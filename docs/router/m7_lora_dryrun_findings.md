# M7 LoRA Dry-Run Findings

## Purpose

M7 tested whether the local `openai/gpt-oss-20b` base model could accept a LoRA adapter and enter a tiny training loop on RTX5080 using the standard Transformers/TRL/PEFT path. This was a dry-run only. It was not intended to evaluate quality.

## Commands Executed

Dataset preview, without model loading or training:

```bash
python scripts/train_router_lora.py --max-samples 2
```

Two-sample dry-run:

```bash
python scripts/train_router_lora.py \
  --run-train \
  --local-files-only \
  --load-strategy mxfp4-auto \
  --max-samples 2 \
  --max-length 128 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-train-epochs 1 \
  --lora-r 2 \
  --lora-alpha 4 \
  --target-modules q_proj \
  --output-dir adapters/dryrun-router-lora-mxfp4-2samples
```

Operational log:

- `logs/lora_dryrun_2samples_001.md` (ignored by git)

## Successful Checks

- Dataset preview worked with `--max-samples 2`.
- The default path did not train without `--run-train`.
- The model loaded with `--local-files-only` and `--load-strategy mxfp4-auto`.
- RTX5080 CUDA was visible.
- No CUDA out-of-memory error occurred before the stop.
- `q_proj` target modules were found: 24 module matches.

## Stop Reason

The dry-run stopped during Trainer initialization because the local gpt-oss model is loaded as MXFP4 quantized weights, and standard Transformers/PEFT training does not support that quantization method.

This means the current RTX5080 local path should treat `openai/gpt-oss-20b` as inference-only unless a separately approved training route is chosen.

## Primary Error

```text
ValueError: The model you are trying to fine-tune is quantized with QuantizationMethod.MXFP4
but that quantization method do not support training.
```

## What Was Not Executed

- The 10-sample dry-run was not executed.
- bnb4 was not tried.
- bf16 was not tried.
- Unsloth was not installed or tried.
- No cloud/API route was used.
- No adapter was saved.
- No full 150-row training was run.
- No package reinstall was performed.

## Safety Update

`scripts/train_router_lora.py` now stops before model loading when `--run-train --load-strategy mxfp4-auto` is requested. Dataset preview without `--run-train` remains available.

The guard message is explicit:

- MXFP4 quantized gpt-oss training is not supported by standard Transformers/PEFT path.
- Use this model as inference-only on RTX5080.
- BF16 training requires much larger VRAM.
- bnb4/Unsloth/cloud routes require separate approval.

## Future Options

Any next route requires separate human approval:

- Keep `openai/gpt-oss-20b` as an inference-only router and improve prompts/eval data.
- Investigate a bnb4-compatible training path on a different model or representation.
- Investigate Unsloth or another specialized local training path.
- Use a larger-VRAM local or cloud GPU for bf16/full-precision adapter training.
- Train a smaller local router model and keep gpt-oss-20b as verifier/judge.
