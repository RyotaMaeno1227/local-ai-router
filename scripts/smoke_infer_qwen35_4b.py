#!/usr/bin/env python3
"""Run a text-only smoke inference check for Qwen/Qwen3.5-4B.

This script may download Qwen/Qwen3.5-4B unless --local-files-only is set. It
does not fine-tune, run LoRA, or save adapter artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval_router import (
    extract_json,
    first_runtime_device,
    generate_once_multimodal,
    load_system_prompt,
    router_messages,
    torch_dtype_value,
)
from validate_router_json import load_json, validate_router_output


DEFAULT_USER_PROMPT = (
    "このFEMソルバの境界条件実装を確認したい。コードは手元にあるので、"
    "どの実行モードと検証項目を選ぶべきかJSON router形式で判断してください。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--processor-name", default=None)
    parser.add_argument("--schema", default="schemas/router_output.schema.json")
    parser.add_argument("--system-prompt-file", default="prompts/router_system_v2.md")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_value(args.torch_dtype),
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def main() -> int:
    args = parse_args()

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "AutoModelForMultimodalLM is not available in the current Transformers install. "
            "Stop here; do not install another library without approval."
        ) from exc

    schema = load_json(Path(args.schema))
    system_prompt = load_system_prompt(args.system_prompt_file)

    processor = AutoProcessor.from_pretrained(
        args.processor_name or args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForMultimodalLM.from_pretrained(args.model_name, **model_kwargs(args))
    model.eval()

    user_prompt = (
        f"User request: {args.prompt}\n\n"
        "Return a router decision JSON object conforming to the required schema."
    )
    final_text, final_channel_found, raw_output_chars = generate_once_multimodal(
        args,
        processor,
        model,
        first_runtime_device(model),
        router_messages(system_prompt, [], user_prompt),
    )

    parsed = extract_json(final_text)
    schema_errors = validate_router_output("qwen35_4b_smoke", parsed, schema) if parsed is not None else []
    result = {
        "model_name": args.model_name,
        "final_channel_found": final_channel_found,
        "raw_output_chars": raw_output_chars,
        "json_valid": parsed is not None,
        "schema_valid": parsed is not None and not schema_errors,
        "schema_errors": schema_errors,
        "final_text": final_text,
        "prediction": parsed,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if parsed is not None else 2


if __name__ == "__main__":
    sys.exit(main())
