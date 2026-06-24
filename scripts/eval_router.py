#!/usr/bin/env python3
"""Evaluate router JSON output against the fixed router schema and eval set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_router_json import load_json, validate_router_output


RISK_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
ALLOWED_RISKS = ("low", "medium", "high", "critical")
ALLOWED_MODES = (
    "local_answer",
    "local_rag",
    "code_exec",
    "external_expert",
    "managed_fusion",
    "self_fusion_lite",
    "web_or_rag_required",
    "request_more_info",
)
REQUIRED_KEYS = (
    "task_type",
    "domain",
    "risk",
    "mode",
    "needed_tools",
    "needed_models",
    "verification",
    "fusion_policy",
    "final_answer_policy",
)
DEFAULT_SCHEMA = Path("schemas/router_output.schema.json")
SYSTEM_PROMPT = f"""Return only one final JSON object for a scientific router decision.
No markdown. No prose outside JSON. No chain-of-thought.
Top-level keys: {", ".join(REQUIRED_KEYS)}
risk enum: {", ".join(ALLOWED_RISKS)}
mode enum: {", ".join(ALLOWED_MODES)}
verification={{"required":bool,"checks":[str],"reason":str}}
fusion_policy={{"enabled":bool,"type":str|null,"reason":str,"panel_size":int|null,"judge_required":bool}}
final_answer_policy={{"format":str,"include_uncertainty":bool,"include_sources":bool}}
Use needed_models=["openai/gpt-oss-20b"]. If inputs are missing, mode="request_more_info".
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default="evals/router_eval_001.jsonl")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--predictions", default=None, help="Optional JSONL with id and output/final_text/prediction.")
    parser.add_argument("--results", "--report", dest="results", default=None, help="Output JSON report path.")
    parser.add_argument("--predictions-out", default=None, help="Optional JSONL path for raw model predictions.")
    parser.add_argument("--markdown-log", default=None, help="Optional Markdown run summary path.")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter path. Do not use for base eval.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="Optional 4-bit loading; off by default.")
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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return rows


def extract_final_message(text: str) -> tuple[str, bool]:
    """Extract final-channel assistant message from gpt-oss Harmony-style output."""
    patterns = [
        r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip(), True
    return text.strip(), False


def extract_json(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalized_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()]
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return []


def expected_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("user", row.get("prompt"))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{row.get('id', '<missing-id>')}: missing user/prompt")
    category = row.get("category", "uncategorized")
    return (
        f"Category: {category}\n"
        f"User request: {prompt}\n\n"
        "Return a router decision JSON object conforming to the required schema."
    )


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}: prediction row missing id")
        raw_output = row.get("raw_output", row.get("output", ""))
        final_text = row.get("final_text")
        final_channel_found = bool(row.get("final_channel_found", False))
        if not isinstance(final_text, str):
            final_text, final_channel_found = extract_final_message(str(raw_output))
        predictions[case_id] = {
            "raw_output": raw_output,
            "final_text": final_text,
            "final_channel_found": final_channel_found,
            "prediction": row.get("prediction", row.get("parsed_json")),
        }
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
            reasoning_effort="low",
        )
    return tokenizer(f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:", return_tensors="pt")


def first_runtime_device(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_predictions(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        "torch_dtype": "auto",
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    q_config = quantization_config(args.load_in_4bit)
    if q_config is not None:
        model_kwargs["quantization_config"] = q_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    outputs: dict[str, dict[str, Any]] = {}
    input_device = first_runtime_device(model)
    for idx, row in enumerate(rows, start=1):
        case_id = row["id"]
        print(f"[{idx}/{len(rows)}] {case_id}", file=sys.stderr, flush=True)
        inputs = build_inputs(tokenizer, expected_prompt(row))
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
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
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        final_text, final_channel_found = extract_final_message(raw_output)
        outputs[case_id] = {
            "raw_output": raw_output,
            "final_text": final_text,
            "final_channel_found": final_channel_found,
            "prediction": extract_json(final_text),
        }
    return outputs


def verification_text(parsed: dict[str, Any]) -> str:
    verification = parsed.get("verification")
    if not isinstance(verification, dict):
        return ""
    pieces = normalized_strings(verification.get("checks"))
    reason = verification.get("reason")
    if isinstance(reason, str):
        pieces.append(reason.lower())
    return " ".join(pieces)


def contains_expected_terms(expected: list[str], actual_text: str) -> bool:
    actual = actual_text.lower()
    return all(term.lower() in actual for term in expected)


def score_case(row: dict[str, Any], raw_prediction: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    parsed = raw_prediction.get("prediction")
    if parsed is None:
        parsed = extract_json(raw_prediction.get("final_text", ""))
    schema_errors = validate_router_output(row["id"], parsed, schema) if parsed is not None else []

    expected_tools = set(normalized_strings(row.get("must_tools")))
    predicted_tools = set(normalized_strings(parsed.get("needed_tools") if isinstance(parsed, dict) else None))
    expected_verification = [str(item) for item in row.get("must_verification", [])]
    predicted_mode = parsed.get("mode") if isinstance(parsed, dict) else None
    predicted_risk = str(parsed.get("risk", "")).lower() if isinstance(parsed, dict) else ""
    min_risk = str(row.get("min_risk", "low")).lower()
    fusion_policy = parsed.get("fusion_policy") if isinstance(parsed, dict) else None
    predicted_fusion_enabled = fusion_policy.get("enabled") if isinstance(fusion_policy, dict) else None

    return {
        "id": row["id"],
        "category": row.get("category"),
        "expected_mode": row.get("expected_mode"),
        "predicted_mode": predicted_mode,
        "min_risk": min_risk,
        "predicted_risk": predicted_risk,
        "json_valid": parsed is not None,
        "schema_valid": parsed is not None and not schema_errors,
        "schema_errors": schema_errors,
        "expected_mode_match": predicted_mode == row.get("expected_mode"),
        "must_tools_contained": expected_tools.issubset(predicted_tools),
        "must_verification_contained": contains_expected_terms(expected_verification, verification_text(parsed))
        if isinstance(parsed, dict)
        else False,
        "risk_underestimated": RISK_ORDER.get(predicted_risk, 0) < RISK_ORDER.get(min_risk, 1),
        "fusion_policy_match": predicted_fusion_enabled == row.get("should_use_fusion"),
        "request_more_info": predicted_mode == "request_more_info",
        "final_channel_found": bool(raw_prediction.get("final_channel_found")),
        "final_text": raw_prediction.get("final_text"),
        "raw_output_chars": len(str(raw_prediction.get("raw_output", ""))),
        "prediction": parsed,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(results),
        "json_valid_count": sum(1 for item in results if item["json_valid"]),
        "schema_valid_count": sum(1 for item in results if item["schema_valid"]),
        "expected_mode_match_count": sum(1 for item in results if item["expected_mode_match"]),
        "must_tools_contained_count": sum(1 for item in results if item["must_tools_contained"]),
        "must_verification_contained_count": sum(1 for item in results if item["must_verification_contained"]),
        "risk_underestimated_count": sum(1 for item in results if item["risk_underestimated"]),
        "fusion_policy_match_count": sum(1 for item in results if item["fusion_policy_match"]),
        "request_more_info_count": sum(1 for item in results if item["request_more_info"]),
        "final_channel_found_count": sum(1 for item in results if item["final_channel_found"]),
    }


def default_results_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("eval_results") / f"router_eval_{stamp}.json"


def write_predictions(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            row = {
                "id": result["id"],
                "category": result.get("category"),
                "final_text": result.get("final_text"),
                "final_channel_found": result.get("final_channel_found"),
                "prediction": result.get("prediction"),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_markdown_log(path: Path, args: argparse.Namespace, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report["metrics"]
    lines = [
        "# Router Base Eval",
        "",
        f"Generated: `{report['meta']['generated_at']}`",
        f"Model: `{args.model_name}`",
        f"Eval file: `{args.eval_file}`",
        f"Schema: `{args.schema}`",
        f"Max cases: `{args.max_cases}`",
        f"Local files only: `{args.local_files_only}`",
        f"Max new tokens: `{args.max_new_tokens}`",
        f"Temperature: `{args.temperature}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        f"Results JSON: `{report['meta']['results_path']}`",
    ]
    predictions_path = report["meta"].get("predictions_path")
    if predictions_path:
        lines.append(f"Predictions JSONL: `{predictions_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_rows = read_jsonl(Path(args.eval_file))
    if args.max_cases is not None:
        eval_rows = eval_rows[: args.max_cases]
    schema = load_json(Path(args.schema))

    if args.predictions:
        predictions = load_predictions(Path(args.predictions))
    else:
        print("No --predictions supplied; loading model for base eval.", file=sys.stderr)
        predictions = generate_predictions(args, eval_rows)

    results = []
    for row in eval_rows:
        case_id = row["id"]
        raw_prediction = predictions.get(
            case_id,
            {"raw_output": "", "final_text": "", "final_channel_found": False, "prediction": None},
        )
        results.append(score_case(row, raw_prediction, schema))

    results_path = Path(args.results) if args.results else default_results_path()
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_name": args.model_name,
            "eval_file": args.eval_file,
            "schema": args.schema,
            "max_cases": args.max_cases,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "local_files_only": args.local_files_only,
            "load_in_4bit": args.load_in_4bit,
            "results_path": str(results_path),
        },
        "metrics": summarize(results),
        "results": results,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.predictions_out:
        predictions_path = Path(args.predictions_out)
        write_predictions(predictions_path, results)
        report["meta"]["predictions_path"] = str(predictions_path)
        results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown_log:
        write_markdown_log(Path(args.markdown_log), args, report)

    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Results: {results_path}")
    if args.predictions_out:
        print(f"Predictions: {args.predictions_out}")
    if args.markdown_log:
        print(f"Markdown log: {args.markdown_log}")


if __name__ == "__main__":
    main()
