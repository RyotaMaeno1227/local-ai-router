#!/usr/bin/env python3
"""Run a short local inference smoke test for the base model.

This script loads model weights and may trigger a Hugging Face download unless
`--local-files-only` is provided and the model already exists in the local cache.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a local scientific AI router/verifier. Output only compact JSON "
    "with keys mode, tools, risk, needs_human_approval, next_action. "
    "Do not include chain-of-thought."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", default="Check whether a FEM solver log should be parsed or rerun.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_inputs(tokenizer: AutoTokenizer, prompt: str) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    text = f"System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
    return tokenizer(text, return_tensors="pt")


def maybe_quantization_config(load_in_4bit: bool) -> Any | None:
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    dtype = torch_dtype(args.dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    quantization_config = maybe_quantization_config(args.load_in_4bit)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    inputs = build_inputs(tokenizer, args.prompt)
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

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    print("\nParsed JSON:")
    print(json.dumps(parsed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
