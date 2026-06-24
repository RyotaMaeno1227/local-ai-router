#!/usr/bin/env python3
"""Run router inference with a LoRA adapter, or baseline with --no-adapter."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a local scientific AI router/verifier. Output only compact JSON "
    "with keys mode, tools, risk, needs_human_approval, next_action, checks, notes. "
    "Allowed risk values: low, medium, high, critical. Do not include chain-of-thought."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path", default="adapters/router-lora-r4")
    parser.add_argument("--prompt", default=None, help="Prompt text. If omitted, stdin is used.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base model only.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def prompt_text(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("No prompt supplied via --prompt or stdin.")
    return text


def quantization_config(load_in_4bit: bool) -> Any | None:
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    q_config = quantization_config(args.load_in_4bit)
    if q_config is not None:
        model_kwargs["quantization_config"] = q_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if not args.no_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer: Any, user_prompt: str) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    text = f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"
    return tokenizer(text, return_tensors="pt")


def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def main() -> None:
    args = parse_args()
    user_prompt = prompt_text(args)
    model, tokenizer = load_model(args)
    inputs = build_inputs(tokenizer, user_prompt)
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(text)

    parsed = extract_json(text)
    if parsed is not None:
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
