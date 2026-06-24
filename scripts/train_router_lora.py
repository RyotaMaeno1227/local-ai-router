#!/usr/bin/env python3
"""LoRA/QLoRA training skeleton for router/verifier SFT.

Default behavior is a no-download dataset preview. Actual training requires
`--run-train`, so invoking this script accidentally will not start fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


SYSTEM_FALLBACK = "You are a local scientific AI router/verifier. Output only compact JSON."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--dataset", default="data/router_sft_001.jsonl")
    parser.add_argument("--output-dir", default="adapters/router-lora-r4")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit SFT rows for a future approved test run.")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--trainer", choices=("trl", "transformers"), default="trl")
    parser.add_argument("--load-strategy", choices=("mxfp4-auto", "bnb4", "bf16"), default="mxfp4-auto")
    parser.add_argument("--model-family", choices=("causal-lm", "multimodal-lm"), default="causal-lm")
    parser.add_argument("--processor-name", default=None, help="Optional processor path/name for multimodal models.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model torch dtype. For Qwen dry-runs use bfloat16.",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training args. Default is disabled.")
    parser.add_argument("--no-4bit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--list-lora-targets", action="store_true", help="Load model, list candidate LoRA modules, and exit.")
    parser.add_argument("--run-train", action="store_true", help="Actually load the model and train.")
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
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no}: missing messages")
            rows.append(row)
    return rows


def fallback_format(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"{role.upper()}: {content}")
    return "\n".join(chunks).strip()


def text_tokenizer(processor_or_tokenizer: Any) -> Any:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def format_messages(processor_or_tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(processor_or_tokenizer, "apply_chat_template"):
        try:
            return processor_or_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return processor_or_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    tokenizer = text_tokenizer(processor_or_tokenizer)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return fallback_format(messages)


def preview_dataset(path: Path, max_samples: int | None = None) -> None:
    rows = read_jsonl(path)
    total_rows = len(rows)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        rows = rows[:max_samples]
    print(f"Dataset: {path}")
    print(f"Examples: {len(rows)}")
    if max_samples is not None:
        print(f"Total examples before --max-samples: {total_rows}")
    first = rows[0]["messages"] if rows else [{"role": "system", "content": SYSTEM_FALLBACK}]
    print("\nFirst formatted example preview:")
    print(fallback_format(first)[:1600])


def quantization_config(load_strategy: str) -> Any | None:
    if load_strategy != "bnb4":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_training_dataset(dataset_path: Path, tokenizer: Any, max_samples: int | None = None) -> Any:
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_messages(tokenizer, example["messages"])}

    remove_columns = [name for name in dataset.column_names if name != "text"]
    return dataset.map(to_text, remove_columns=remove_columns)


def optimizer_name(args: argparse.Namespace) -> str:
    return "paged_adamw_8bit" if args.load_strategy == "bnb4" else "adamw_torch"


def torch_dtype_value(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def model_torch_dtype(args: argparse.Namespace) -> Any:
    if args.torch_dtype != "auto":
        return torch_dtype_value(args.torch_dtype)
    if args.load_strategy == "mxfp4-auto":
        return "auto"
    if args.load_strategy == "bf16":
        return torch.bfloat16
    return "auto"


def ensure_target_modules_exist(model: Any, target_modules: list[str]) -> None:
    counts = {target: 0 for target in target_modules}
    for name, _module in model.named_modules():
        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        for target in target_modules:
            if leaf_name == target or name.endswith(target):
                counts[target] += 1

    missing = [target for target, count in counts.items() if count == 0]
    print("Target module matches:")
    for target, count in counts.items():
        print(f"  {target}: {count}")
    if missing:
        raise ValueError(f"target_modules not found: {', '.join(missing)}")


def list_lora_target_modules(model: Any, target_modules: list[str]) -> None:
    candidates = list(dict.fromkeys([*target_modules, "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    matches: dict[str, list[str]] = {candidate: [] for candidate in candidates}
    for name, _module in model.named_modules():
        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        for candidate in candidates:
            if leaf_name == candidate or name.endswith(candidate):
                matches[candidate].append(name)

    print("LoRA target candidate module counts:")
    for candidate, names in matches.items():
        print(f"  {candidate}: {len(names)}")
        for name in names[:20]:
            print(f"    - {name}")
        if len(names) > 20:
            print(f"    ... {len(names) - 20} more")


def print_trainable_parameters(model: Any) -> None:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = 100 * trainable / total if total else 0.0
    print(f"LoRA trainable parameters: {trainable:,} / {total:,} ({ratio:.6f}%)")
    if trainable == 0:
        raise RuntimeError("No trainable parameters found after LoRA insertion.")


def add_nan_guard_callback(trainer: Any) -> None:
    from transformers import TrainerCallback

    class NanGuardCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs or "loss" not in logs:
                return
            loss = float(logs["loss"])
            if not math.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected: {loss}")

    trainer.add_callback(NanGuardCallback())


def ensure_training_strategy_allowed(args: argparse.Namespace) -> None:
    if args.run_train and args.load_strategy == "mxfp4-auto" and "gpt-oss" in args.model_name.lower():
        raise SystemExit(
            "MXFP4 quantized gpt-oss training is not supported by standard Transformers/PEFT path.\n"
            "Use this model as inference-only on RTX5080.\n"
            "BF16 training requires much larger VRAM.\n"
            "bnb4/Unsloth/cloud routes require separate approval."
        )


def ensure_tokenizer_padding(processor_or_tokenizer: Any) -> None:
    tokenizer = text_tokenizer(processor_or_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def build_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.no_4bit and args.load_strategy == "bnb4":
        raise ValueError("--no-4bit is incompatible with --load-strategy bnb4")

    q_config = quantization_config(args.load_strategy)
    if q_config is not None:
        model_kwargs["quantization_config"] = q_config
    else:
        model_kwargs["torch_dtype"] = model_torch_dtype(args)

    print(f"Loading model with strategy: {args.load_strategy}")
    print(f"Model kwargs: torch_dtype={model_kwargs.get('torch_dtype')}, device_map={model_kwargs.get('device_map')}")

    if args.model_family == "multimodal-lm":
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
        ensure_tokenizer_padding(processor)
        model = AutoModelForMultimodalLM.from_pretrained(args.model_name, **model_kwargs)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        ensure_tokenizer_padding(processor)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if q_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, processor


def train_with_trl(args: argparse.Namespace, model: Any, tokenizer: Any, train_dataset: Any, lora_config: Any) -> Any:
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        training_args = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=args.max_length,
            packing=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="epoch",
            bf16=args.bf16,
            fp16=False,
            report_to="none",
            gradient_checkpointing=True,
            optim=optimizer_name(args),
            do_train=True,
        )
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
    except TypeError:
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="epoch",
            bf16=args.bf16,
            fp16=False,
            report_to="none",
            gradient_checkpointing=True,
            optim=optimizer_name(args),
        )
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=args.max_length,
            packing=False,
        )


def train_with_transformers(args: argparse.Namespace, model: Any, tokenizer: Any, train_dataset: Any) -> Any:
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=False,
        report_to="none",
        gradient_checkpointing=True,
        optim=optimizer_name(args),
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    ensure_training_strategy_allowed(args)

    if not args.run_train and not args.list_lora_targets:
        preview_dataset(dataset_path, max_samples=args.max_samples)
        print("\nTraining was not started.")
        print("Add --run-train only after baseline eval and human approval.")
        return

    from peft import LoraConfig, TaskType

    model, processor = build_model_and_processor(args)
    tokenizer = text_tokenizer(processor)

    if args.list_lora_targets:
        list_lora_target_modules(model, args.target_modules)
        print("\nTraining was not started.")
        return

    ensure_target_modules_exist(model, args.target_modules)
    train_dataset = load_training_dataset(dataset_path, processor, max_samples=args.max_samples)
    print(f"Training examples loaded: {len(train_dataset)}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    if args.trainer == "trl" and args.model_family != "multimodal-lm":
        trainer = train_with_trl(args, model, tokenizer, train_dataset, lora_config)
        print_trainable_parameters(trainer.model)
    else:
        from peft import get_peft_model

        if args.trainer == "trl" and args.model_family == "multimodal-lm":
            print("Using transformers Trainer for multimodal-lm text-only dry-run.")
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        trainer = train_with_transformers(args, model, tokenizer, train_dataset)

    add_nan_guard_callback(trainer)
    trainer.train()
    trainer.save_model(args.output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.output_dir)
    else:
        tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
