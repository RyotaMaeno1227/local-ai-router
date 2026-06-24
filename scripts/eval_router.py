#!/usr/bin/env python3
"""Evaluate router JSON output against the fixed router schema and eval set."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--save-predictions", action="store_true", help="Save final-only predictions JSONL.")
    parser.add_argument("--markdown-log", default=None, help="Optional Markdown run summary path.")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter path. Do not use for base eval.")
    parser.add_argument("--system-prompt-file", default=None, help="Optional router system prompt file.")
    parser.add_argument("--fewshot-file", default=None, help="Optional JSONL few-shot messages file.")
    parser.add_argument("--prompt-version", default="base", help="Prompt version label for reporting.")
    parser.add_argument("--repair-json-once", action="store_true", help="Retry once when output is invalid JSON/schema.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-family", choices=("auto", "causal-lm", "multimodal-lm"), default="auto")
    parser.add_argument("--model-class", choices=("auto", "causal-lm", "multimodal-lm"), default="auto")
    parser.add_argument("--processor-name", default=None, help="Optional processor path/name for multimodal models.")
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model torch dtype. Keep auto unless a model-specific run requires otherwise.",
    )
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


def load_system_prompt(path: str | None) -> str:
    if not path:
        return SYSTEM_PROMPT
    return Path(path).read_text(encoding="utf-8").strip()


def assistant_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_fewshot_messages(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    messages: list[dict[str, str]] = []
    for row_idx, row in enumerate(read_jsonl(Path(path)), start=1):
        if isinstance(row.get("messages"), list):
            for message in row["messages"]:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{row_idx}: message must be an object")
                role = message.get("role")
                if role == "system":
                    continue
                if role not in {"user", "assistant"}:
                    raise ValueError(f"{path}:{row_idx}: unsupported few-shot role {role!r}")
                content = message.get("content")
                if role == "assistant":
                    content = assistant_json_string(content)
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{row_idx}: few-shot {role} content must be non-empty")
                messages.append({"role": role, "content": content})
            continue

        user = row.get("user")
        assistant = row.get("assistant", row.get("output"))
        if not isinstance(user, str) or not user:
            raise ValueError(f"{path}:{row_idx}: few-shot row missing user")
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant_json_string(assistant)})
    return messages


def extract_final_message(text: str) -> tuple[str, bool]:
    """Extract final-channel assistant message from gpt-oss Harmony-style output."""
    patterns = [
        r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
        r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*)$",
        r"<\|channel\|>final<\|message\|>(.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip(), True
    return text.strip(), False


def strip_internal_reasoning(text: str) -> str:
    stripped = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", maxsplit=1)[-1].strip()
    return stripped


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
            "repair_attempted": bool(row.get("repair_attempted", False)),
            "repair_success": bool(row.get("repair_success", False)),
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


def torch_dtype_value(name: str) -> Any:
    if name == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolved_model_family(args: argparse.Namespace) -> str:
    if args.model_class != "auto":
        return str(args.model_class)
    if args.model_family != "auto":
        return str(args.model_family)
    return "causal-lm"


def patch_hub_kernel_trust(args: argparse.Namespace) -> None:
    if not args.trust_remote_code:
        return
    try:
        from transformers import __version__ as transformers_version
        from transformers.integrations import hub_kernels
        from kernels import get_kernel as kernels_get_kernel
        from kernels import get_local_kernel
    except Exception:
        return

    def local_kernel_snapshot(repo_id: str) -> Path | None:
        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if cache_root:
            hub_root = Path(cache_root)
        else:
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            hub_root = hf_home / "hub"
        repo_cache = hub_root / f"kernels--{repo_id.replace('/', '--')}"
        snapshots = repo_cache / "snapshots"
        ref_path = repo_cache / "refs" / "main"
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            candidate = snapshots / revision
            if candidate.exists():
                return candidate
        if snapshots.exists():
            candidates = [path for path in snapshots.iterdir() if path.is_dir()]
            if candidates:
                return max(candidates, key=lambda path: path.stat().st_mtime)
        return None

    def get_kernel_with_trust(
        kernel_name: str,
        revision: str | None = None,
        version: int | str | None = None,
        allow_all_kernels: bool = False,
    ) -> Any:
        repo_parent = kernel_name.split("/")[0]
        if repo_parent != "kernels-community" and not allow_all_kernels:
            raise ValueError("Use allow_all_kernels=True for kernels outside kernels-community.")
        if args.local_files_only:
            snapshot = local_kernel_snapshot(kernel_name)
            if snapshot is None:
                raise FileNotFoundError(f"No local cached kernel snapshot found for {kernel_name}")
            return get_local_kernel(snapshot)
        user_agent = {"framework": "transformers", "version": transformers_version, "repo_id": kernel_name}
        return kernels_get_kernel(
            kernel_name,
            revision=revision,
            version=version,
            user_agent=user_agent,
            trust_remote_code=True,
        )

    hub_kernels.get_kernel = get_kernel_with_trust


def build_inputs(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        )
    text = "\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)
    return tokenizer(f"{text}\nAssistant:", return_tensors="pt")


def build_multimodal_inputs(processor: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            try:
                rendered = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
            except TypeError:
                rendered = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            try:
                return processor(text=[rendered], return_tensors="pt")
            except TypeError:
                return processor(rendered, return_tensors="pt")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

    text = "\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)
    try:
        return processor(text=[f"{text}\nAssistant:"], return_tensors="pt")
    except TypeError:
        return processor(f"{text}\nAssistant:", return_tensors="pt")


def text_decoder(processor_or_tokenizer: Any) -> Any:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def token_id(decoder: Any, name: str) -> int | None:
    value = getattr(decoder, name, None)
    if isinstance(value, int):
        return value
    return None


def router_messages(system_prompt: str, fewshot_messages: list[dict[str, str]], user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        *fewshot_messages,
        {"role": "user", "content": user_prompt},
    ]


def repair_messages(
    system_prompt: str,
    original_user_prompt: str,
    final_text: str,
    schema_errors: list[str],
) -> list[dict[str, str]]:
    repair_user = (
        "The previous final answer failed JSON parsing or router schema validation.\n"
        "Repair only the router JSON object. Do not answer the task itself. Do not include markdown, comments, or chain-of-thought.\n\n"
        f"Original router request:\n{original_user_prompt}\n\n"
        f"Schema errors:\n{json.dumps(schema_errors[:12], ensure_ascii=False)}\n\n"
        f"Previous final answer:\n{final_text}\n\n"
        "Return exactly one valid JSON object with all required router schema keys."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_user},
    ]


def first_runtime_device(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_once(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    input_device: Any,
    messages: list[dict[str, str]],
) -> tuple[str, bool, int]:
    import torch

    inputs = build_inputs(tokenizer, messages)
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
    del inputs, output_ids, new_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_text, final_channel_found, len(raw_output)


def generate_once_multimodal(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    input_device: Any,
    messages: list[dict[str, str]],
) -> tuple[str, bool, int]:
    import torch

    inputs = build_multimodal_inputs(processor, messages)
    inputs = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    decoder = text_decoder(processor)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
    }
    pad_token_id = token_id(decoder, "pad_token_id")
    eos_token_id = token_id(decoder, "eos_token_id")
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_ids = inputs.get("input_ids")
    input_length = input_ids.shape[-1] if input_ids is not None else 0
    new_tokens = output_ids[0][input_length:]
    raw_output = strip_internal_reasoning(decoder.decode(new_tokens, skip_special_tokens=True).strip())
    final_text, final_channel_found = extract_final_message(raw_output)
    del inputs, output_ids, new_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_text, final_channel_found, len(raw_output)


def schema_errors_for(case_id: str, parsed: dict[str, Any] | None, schema: dict[str, Any]) -> list[str]:
    if parsed is None:
        return [f"{case_id}: output is not valid JSON object"]
    return validate_router_output(case_id, parsed, schema)


def generate_predictions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    system_prompt: str,
    fewshot_messages: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_hub_kernel_trust(args)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_value(args.torch_dtype),
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    q_config = quantization_config(args.load_in_4bit)
    if q_config is not None:
        model_kwargs["quantization_config"] = q_config

    model_family = resolved_model_family(args)
    if model_family == "multimodal-lm":
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "AutoModelForMultimodalLM is not available in the current Transformers install. "
                "Stop here; do not install another library without approval."
            ) from exc
        processor = AutoProcessor.from_pretrained(
            args.processor_name or args.model_name,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForMultimodalLM.from_pretrained(args.model_name, **model_kwargs)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        processor = None
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
        user_prompt = expected_prompt(row)
        messages = router_messages(system_prompt, fewshot_messages, user_prompt)
        if model_family == "multimodal-lm":
            final_text, final_channel_found, raw_output_chars = generate_once_multimodal(
                args,
                processor,
                model,
                input_device,
                messages,
            )
        else:
            final_text, final_channel_found, raw_output_chars = generate_once(
                args,
                tokenizer,
                model,
                input_device,
                messages,
            )
        prediction = extract_json(final_text)
        schema_errors = schema_errors_for(case_id, prediction, schema)
        repair_attempted = False
        repair_success = False

        if schema_errors and args.repair_json_once:
            repair_attempted = True
            print(f"[{idx}/{len(rows)}] {case_id} repair_json_once", file=sys.stderr, flush=True)
            repair_prompt = repair_messages(system_prompt, user_prompt, final_text, schema_errors)
            if model_family == "multimodal-lm":
                repaired_text, repaired_final_channel_found, repaired_chars = generate_once_multimodal(
                    args,
                    processor,
                    model,
                    input_device,
                    repair_prompt,
                )
            else:
                repaired_text, repaired_final_channel_found, repaired_chars = generate_once(
                    args,
                    tokenizer,
                    model,
                    input_device,
                    repair_prompt,
                )
            repaired_prediction = extract_json(repaired_text)
            repaired_errors = schema_errors_for(case_id, repaired_prediction, schema)
            repair_success = not repaired_errors
            final_text = repaired_text
            final_channel_found = repaired_final_channel_found
            raw_output_chars = repaired_chars
            prediction = repaired_prediction
            schema_errors = repaired_errors

        outputs[case_id] = {
            "final_text": final_text,
            "final_channel_found": final_channel_found,
            "prediction": prediction,
            "schema_errors": schema_errors,
            "repair_attempted": repair_attempted,
            "repair_success": repair_success,
            "raw_output_chars": raw_output_chars,
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
        "repair_attempted": bool(raw_prediction.get("repair_attempted")),
        "repair_success": bool(raw_prediction.get("repair_success")),
        "final_text": raw_prediction.get("final_text"),
        "raw_output_chars": int(raw_prediction.get("raw_output_chars", len(str(raw_prediction.get("raw_output", ""))))),
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
        "repair_attempted_count": sum(1 for item in results if item["repair_attempted"]),
        "repair_success_count": sum(1 for item in results if item["repair_success"]),
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
                "repair_attempted": result.get("repair_attempted"),
                "repair_success": result.get("repair_success"),
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
        f"Model family: `{resolved_model_family(args)}`",
        f"Model class: `{args.model_class}`",
        f"Processor name: `{args.processor_name}`",
        f"Attention implementation: `{args.attn_implementation}`",
        f"Torch dtype: `{args.torch_dtype}`",
        f"Prompt version: `{args.prompt_version}`",
        f"System prompt file: `{args.system_prompt_file}`",
        f"Few-shot file: `{args.fewshot_file}`",
        f"Repair JSON once: `{args.repair_json_once}`",
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
    system_prompt = load_system_prompt(args.system_prompt_file)
    fewshot_messages = load_fewshot_messages(args.fewshot_file)

    if args.predictions:
        predictions = load_predictions(Path(args.predictions))
    else:
        print("No --predictions supplied; loading model for base eval.", file=sys.stderr)
        predictions = generate_predictions(args, eval_rows, schema, system_prompt, fewshot_messages)

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
            "prompt_version": args.prompt_version,
            "system_prompt_file": args.system_prompt_file,
            "fewshot_file": args.fewshot_file,
            "fewshot_message_count": len(fewshot_messages),
            "repair_json_once": args.repair_json_once,
            "results_path": str(results_path),
        },
        "metrics": summarize(results),
        "results": results,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    predictions_out = args.predictions_out
    if args.save_predictions and predictions_out is None:
        predictions_out = str(results_path.with_name(f"{results_path.stem}_predictions.jsonl"))

    if predictions_out:
        predictions_path = Path(predictions_out)
        write_predictions(predictions_path, results)
        report["meta"]["predictions_path"] = str(predictions_path)
        results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown_log:
        write_markdown_log(Path(args.markdown_log), args, report)

    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Results: {results_path}")
    if predictions_out:
        print(f"Predictions: {predictions_out}")
    if args.markdown_log:
        print(f"Markdown log: {args.markdown_log}")


if __name__ == "__main__":
    main()
