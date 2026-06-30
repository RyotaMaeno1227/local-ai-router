# Qwen3.5 4B Router LoRA v002-small Training

M17 trained one Qwen3.5 4B LoRA adapter for one epoch on the frozen 90-row
router SFT v002 candidate. This was the only training run approved for M17.

## Command

```bash
conda run --no-capture-output -n gptoss20b python scripts/train_router_lora.py \
  --run-train \
  --model-name Qwen/Qwen3.5-4B \
  --model-family multimodal-lm \
  --processor-name Qwen/Qwen3.5-4B \
  --dataset data/router_sft_v002_candidate.jsonl \
  --output-dir adapters/qwen35-4b-router-lora-v002-small \
  --max-length 256 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 1 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules q_proj v_proj \
  --load-strategy bf16 \
  --torch-dtype bfloat16 \
  --bf16 \
  --local-files-only
```

## Dataset And Configuration

- dataset: `data/router_sft_v002_candidate.jsonl`
- rows: 90
- dataset SHA-256: `f451b8007cf9992ab4310625bd5d01078049e5bcb919d3ef6647b94873ec2332`
- base model and processor: `Qwen/Qwen3.5-4B`
- model family: multimodal LM with text-only SFT
- maximum sequence length: 256
- per-device batch size: 1
- gradient accumulation: 4
- epochs: 1
- dtype: bfloat16
- LoRA: `r=4`, `alpha=8`, dropout `0.05`
- target modules: `q_proj`, `v_proj` (8 matches each)
- local files only: enabled

## Result

- trainable parameters: 458,752 / 4,539,724,288 (0.010105%)
- optimizer steps: 23
- runtime: 85.0 seconds
- final aggregate train loss: 2.637
- final logged step loss: 2.412
- CUDA OOM: no
- NaN or non-finite loss: no
- adapter save: succeeded
- adapter path: `adapters/qwen35-4b-router-lora-v002-small`
- operational log: `logs/qwen35_lora_v002_small_train_001.md`

The adapter directory is git-ignored and is not part of the repository
commit. No 150-row training, Qwen3.5 9B experiment, GPT-OSS fine-tuning,
dependency installation, or external API connection was performed.
