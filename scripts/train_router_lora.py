#!/usr/bin/env python3
"""LoRA/QLoRA training skeleton for router/verifier SFT.

Default behavior is a no-download dataset preview. Actual training requires
`--run-train`, so invoking this script accidentally will not start fine-tuning.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes 4-bit QLoRA loading.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files.")
    parser.add_argument("--trust-remote-code", action="store_true")
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


def format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
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


def build_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    q_config = quantization_config(load_in_4bit=not args.no_4bit)
    if q_config is not None:
        model_kwargs["quantization_config"] = q_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if q_config is not None:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train_with_trl(args: argparse.Namespace, model: Any, tokenizer: Any, train_dataset: Any, lora_config: Any) -> Any:
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        training_args = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_seq_length=args.max_length,
            packing=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=1,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to="none",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
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
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to="none",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
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
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
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

    if not args.run_train:
        preview_dataset(dataset_path, max_samples=args.max_samples)
        print("\nTraining was not started.")
        print("Add --run-train only after baseline eval and human approval.")
        return

    from peft import LoraConfig, TaskType

    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset = load_training_dataset(dataset_path, tokenizer, max_samples=args.max_samples)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    if args.trainer == "trl":
        trainer = train_with_trl(args, model, tokenizer, train_dataset, lora_config)
    else:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
        trainer = train_with_transformers(args, model, tokenizer, train_dataset)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
