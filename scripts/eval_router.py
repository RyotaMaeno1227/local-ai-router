#!/usr/bin/env python3
"""Evaluate router JSON output against minimal routing expectations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RISK_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SYSTEM_PROMPT = (
    "You are a local scientific AI router/verifier. Output only compact JSON "
    "with keys task_type, domain, risk, mode, needed_tools, needed_models, "
    "verification, fusion_policy, final_answer_policy. Allowed risk values: "
    "low, medium, high, critical. Do not include chain-of-thought."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default="evals/router_eval_001.jsonl")
    parser.add_argument("--predictions", default=None, help="Optional JSONL with id and output or prediction.")
    parser.add_argument("--report", default=None, help="Optional report JSON path.")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def extract_json(text: str) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    decoder = json.JSONDecoder()
    for idx, char in enumerate(str(text)):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(str(text)[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalized_tools(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()}
    if isinstance(value, list):
        return {str(item).strip().lower() for item in value}
    return set()


def prompt_for_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt", row.get("user"))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{row.get('id', '<missing-id>')}: missing prompt/user")
    return prompt


def load_predictions(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    predictions: dict[str, Any] = {}
    for row in rows:
        case_id = row.get("id")
        if not case_id:
            raise ValueError(f"{path}: prediction row missing id")
        predictions[case_id] = row.get("prediction", row.get("output", row))
    return predictions


def quantization_config(load_in_4bit: bool) -> Any | None:
    if not load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_inputs(tokenizer: Any, user_prompt: str) -> dict[str, Any]:
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
    return tokenizer(f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:", return_tensors="pt")


def generate_predictions(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    outputs: dict[str, str] = {}
    first_device = next(model.parameters()).device
    for row in rows:
        inputs = build_inputs(tokenizer, prompt_for_row(row))
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
        outputs[row["id"]] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return outputs


def score_case(row: dict[str, Any], raw_prediction: Any) -> dict[str, Any]:
    parsed = extract_json(raw_prediction)
    result: dict[str, Any] = {
        "id": row["id"],
        "json_valid": parsed is not None,
        "expected_mode_match": False,
        "must_tools_contained": False,
        "risk_underestimated": True,
        "prediction": parsed,
    }
    if parsed is None:
        result["error"] = "prediction is not valid JSON"
        return result

    expected_mode = row.get("expected_mode")
    result["expected_mode_match"] = parsed.get("mode") == expected_mode

    expected_tools = normalized_tools(row.get("must_tools"))
    predicted_tools = normalized_tools(parsed.get("needed_tools", parsed.get("tools")))
    result["must_tools_contained"] = expected_tools.issubset(predicted_tools)

    min_risk = str(row.get("min_risk", "low")).lower()
    predicted_risk = str(parsed.get("risk", "")).lower()
    result["risk_underestimated"] = RISK_ORDER.get(predicted_risk, 0) < RISK_ORDER.get(min_risk, 1)
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}
    return {
        "total": total,
        "json_valid": sum(1 for item in results if item["json_valid"]),
        "expected_mode_match": sum(1 for item in results if item["expected_mode_match"]),
        "must_tools_contained": sum(1 for item in results if item["must_tools_contained"]),
        "risk_underestimated": sum(1 for item in results if item["risk_underestimated"]),
    }


def default_report_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("logs") / f"eval_router_{stamp}.json"


def main() -> None:
    args = parse_args()
    eval_rows = read_jsonl(Path(args.eval_file))

    if args.predictions:
        predictions = load_predictions(Path(args.predictions))
    else:
        print("No --predictions supplied; loading model to generate baseline predictions.", file=sys.stderr)
        predictions = generate_predictions(args, eval_rows)

    results = []
    for row in eval_rows:
        case_id = row["id"]
        if case_id not in predictions:
            results.append(
                {
                    "id": case_id,
                    "json_valid": False,
                    "expected_mode_match": False,
                    "must_tools_contained": False,
                    "risk_underestimated": True,
                    "error": "missing prediction",
                    "prediction": None,
                }
            )
            continue
        results.append(score_case(row, predictions[case_id]))

    report = {"summary": summarize(results), "results": results}
    report_path = Path(args.report) if args.report else default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report: {report_path}")

    summary = report["summary"]
    total = summary.get("total", 0)
    if summary.get("risk_underestimated", 0) > 0:
        raise SystemExit(2)
    if (
        summary.get("json_valid", 0) < total
        or summary.get("expected_mode_match", 0) < total
        or summary.get("must_tools_contained", 0) < total
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
