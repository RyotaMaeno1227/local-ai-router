# gpt-oss-20b Local Router/Verifier Starter

WSL2 Ubuntu 上で `openai/gpt-oss-20b` をローカル科学系 AI 指揮者の router/verifier として試すための最小プロジェクトです。
最初の範囲は環境確認、短い推論確認、router/verifier SFT データ形式、LoRA/QLoRA 学習スクリプト雛形、評価スクリプト雛形までです。

本格 fine-tuning はまだ実行しません。必ず baseline eval を取り、人間が承認してから `--run-train` を付けてください。

## Directory

```text
~/local-ai/gpt-oss-20b/
  data/router_sft_001.jsonl
  evals/router_eval_001.jsonl
  scripts/check_env.py
  scripts/smoke_infer.py
  scripts/train_router_lora.py
  scripts/infer_router_lora.py
  scripts/eval_router.py
  adapters/
  logs/
```

## Setup

Windows 側や WSL の OS 設定はこのプロジェクトでは変更しません。作業場所は必ず WSL 側の `~/local-ai/gpt-oss-20b` です。`/mnt/c/...` 配下には作らないでください。

```bash
cd ~/local-ai/gpt-oss-20b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

PyTorch は WSL2 から見える CUDA に合う公式 wheel を選んで入れてください。例:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade transformers accelerate peft trl bitsandbytes datasets safetensors
```

このプロジェクトは API キーや認証情報を要求しません。モデル取得に認証が必要な環境では、Hugging Face 側のローカル認証を別途管理してください。

## Run Order

1. 環境確認:

```bash
python scripts/check_env.py
```

2. モデルをダウンロードしてよいことを確認してから、短い smoke inference:

```bash
python scripts/smoke_infer.py --model-name openai/gpt-oss-20b --load-in-4bit
```

既にローカルキャッシュにあるモデルだけを使う場合:

```bash
python scripts/smoke_infer.py --model-name openai/gpt-oss-20b --load-in-4bit --local-files-only
```

3. fine-tuning 前に baseline eval を取る:

```bash
python scripts/eval_router.py \
  --eval-file evals/router_eval_001.jsonl \
  --model-name openai/gpt-oss-20b \
  --load-in-4bit
```

`eval_router.py` は JSON 妥当性、`expected_mode` 一致、`must_tools` 包含、risk 過小評価を確認します。risk 過小評価がある場合は終了コード `2` で終了します。

4. 学習設定を確認するだけ:

```bash
python scripts/train_router_lora.py --dataset data/router_sft_001.jsonl
```

5. 人間が承認した後だけ、LoRA/QLoRA 学習を開始:

```bash
python scripts/train_router_lora.py \
  --dataset data/router_sft_001.jsonl \
  --output-dir adapters/router-lora-r4 \
  --model-name openai/gpt-oss-20b \
  --max-length 512 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 4 \
  --target-modules q_proj v_proj \
  --run-train
```

6. adapter 推論:

```bash
python scripts/infer_router_lora.py \
  --model-name openai/gpt-oss-20b \
  --adapter-path adapters/router-lora-r4 \
  --load-in-4bit \
  --prompt "Verify whether a solver convergence claim needs numerical checks."
```

7. adapter eval:

```bash
python scripts/eval_router.py \
  --eval-file evals/router_eval_001.jsonl \
  --model-name openai/gpt-oss-20b \
  --adapter-path adapters/router-lora-r4 \
  --load-in-4bit
```

## Data Format

SFT データは `messages` 形式の JSONL です。assistant の内容は `schemas/router_output.schema.json` に従う JSON 文字列のみで、chain-of-thought は含めません。

```json
{"id":"...","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{\"task_type\":\"matrix_property_check\",\"domain\":\"FEM basic\",\"risk\":\"medium\",\"mode\":\"code_exec\",\"needed_tools\":[\"python\"],\"needed_models\":[\"openai/gpt-oss-20b\"],\"verification\":{\"required\":true,\"checks\":[\"symmetry check\"],\"reason\":\"deterministic check required\"},\"fusion_policy\":{\"enabled\":false,\"type\":null,\"reason\":\"Single local route is sufficient.\",\"panel_size\":null,\"judge_required\":false},\"final_answer_policy\":{\"format\":\"json_only\",\"include_uncertainty\":true,\"include_sources\":false}}"}]}
```

評価データは期待値を持つ JSONL です。M4では `evals/router_eval_001.jsonl` を50件へ拡張し、カテゴリ内訳は FEM基礎、非線形FEM、接触解析、コード確認、論文・新規性確認を各10件にしています。

```json
{"id":"...","category":"FEM基礎","user":"...","expected_mode":"code_exec","min_risk":"medium","must_tools":["python"],"must_verification":["symmetry check"],"should_use_fusion":false,"notes":"synthetic"}
```

eval JSONL の検証:

```bash
python scripts/validate_eval_jsonl.py evals/router_eval_001.jsonl
```

router 出力の基本フィールド:

```json
{
  "task_type": "matrix_property_check",
  "domain": "FEM basic",
  "risk": "low | medium | high | critical",
  "mode": "local_answer | local_rag | code_exec | external_expert | managed_fusion | self_fusion_lite | web_or_rag_required | request_more_info",
  "needed_tools": ["python"],
  "needed_models": ["openai/gpt-oss-20b"],
  "verification": {"required": true, "checks": ["short check"], "reason": "short reason"},
  "fusion_policy": {"enabled": false, "type": null, "reason": "short reason", "panel_size": null, "judge_required": false},
  "final_answer_policy": {"format": "json_only", "include_uncertainty": true, "include_sources": false}
}
```

## OOM Handling

RTX5080 16GB 想定の初期設定は控えめです。OOM した場合は次を順番に試してください。

- `--max-length 512` から `--max-length 256` に下げる。
- `--per-device-train-batch-size 1` を維持し、`--gradient-accumulation-steps` を下げる。
- `--load-in-4bit` または学習側の既定 QLoRA を使う。
- `--max-new-tokens` を短くする。
- 他の GPU プロセスを停止し、`nvidia-smi` で VRAM を確認する。
- 必要なら `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付けて再試行する。

## Safety Notes

- 知識注入ではなく router/verifier 挙動の安定化が目的です。
- chain-of-thought を教師データに含めません。
- 出力は JSON router 形式を基本にします。
- 大容量モデルのダウンロード、baseline eval、学習は、それぞれ実行前に人間が確認してください。
- 本格 fine-tuning は必ず人間の承認後に行ってください。

## Environment Check Notes

`logs/env_check_001.md` の確認結果:

- `nvidia-smi` は成功し、RTX5080 16GB が WSL2 から見えています。
- `python --version` は Python 3.13.2 でした。
- `pip --version` は Python 3.10 側の user site-packages を指していました。
- `python scripts/check_env.py` は `No module named 'torch'` で失敗しました。
- `torch`, `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `datasets` は未導入です。

次に必要な人間作業:

- `cd ~/local-ai/gpt-oss-20b` で作業する。
- `python3.12 -m venv .venv` と `source .venv/bin/activate` で venv を作成・有効化する。
- bare `pip` ではなく `python -m pip` を使い、Python と pip の対応を揃える。
- venv 有効化後に PyTorch CUDA wheel と Transformers/TRL/PEFT/bitsandbytes/datasets を導入する。
- 導入後に `python scripts/check_env.py` を再実行する。

## Venv Discipline

今後は必ず `cd ~/local-ai/gpt-oss-20b` の後に `source .venv/bin/activate` してから作業してください。
パッケージ導入は `pip install` ではなく、常に `python -m pip install` を使ってください。
Torch を入れる前に、`python` と `python -m pip` が同じ `.venv` を指していることを確認してください。

確認コマンド:

```bash
which python
python --version
which pip
pip --version
python -m pip --version
python -c "import sys; print(sys.executable); print(sys.version)"
```

`python3.12` が使えない場合、人間が次を実施してください:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

## Router Schema Validation

M3 fixes the router output contract in `schemas/router_output.schema.json`.
Required top-level keys are `task_type`, `domain`, `risk`, `mode`, `needed_tools`, `needed_models`, `verification`, `fusion_policy`, and `final_answer_policy`.
The validator uses only the Python standard library and does not load the model.

Validate a single router JSON file:

```bash
python scripts/validate_router_json.py path/to/router_output.json
```

Validate a JSONL file with one router output per line:

```bash
python scripts/validate_router_json.py --mode jsonl path/to/router_outputs.jsonl
```

Validate SFT `messages` JSONL by checking each assistant `content` JSON string:

```bash
python scripts/validate_router_json.py --mode sft-jsonl path/to/router_sft.jsonl
```

M4 updates `data/router_sft_001.jsonl` to the fixed router schema and expands `evals/router_eval_001.jsonl` to 50 synthetic eval rows.

## Base Router Eval

M5 evaluates the unfine-tuned `openai/gpt-oss-20b` base model against the 50-row router eval set. Use the `gptoss20b` conda environment and prefer `--local-files-only` to avoid model downloads.

Smoke eval with 3 cases:

```bash
conda run -n gptoss20b python scripts/eval_router.py \
  --eval-file evals/router_eval_001.jsonl \
  --schema schemas/router_output.schema.json \
  --model-name openai/gpt-oss-20b \
  --max-cases 3 \
  --local-files-only \
  --max-new-tokens 256 \
  --temperature 0 \
  --results eval_results/base_eval_smoke_001.json \
  --markdown-log logs/base_eval_smoke_001.md
```

Full 50-case base eval:

```bash
conda run -n gptoss20b python scripts/eval_router.py \
  --eval-file evals/router_eval_001.jsonl \
  --schema schemas/router_output.schema.json \
  --model-name openai/gpt-oss-20b \
  --local-files-only \
  --max-new-tokens 256 \
  --temperature 0 \
  --results eval_results/base_eval_001.json \
  --predictions-out eval_results/base_predictions_001.jsonl \
  --markdown-log logs/base_eval_001.md
```

Saved result files:

- `eval_results/base_eval_smoke_001.json`: 3-case smoke metrics and per-case details.
- `eval_results/base_eval_001.json`: 50-case metrics and per-case details.
- `eval_results/base_predictions_001.jsonl`: final-channel model predictions for the 50-case run.
- `logs/base_eval_smoke_001.md` and `logs/base_eval_001.md`: operational logs, ignored by git.

M5 base eval result:

```json
{
  "total": 50,
  "json_valid_count": 50,
  "schema_valid_count": 44,
  "expected_mode_match_count": 16,
  "must_tools_contained_count": 9,
  "must_verification_contained_count": 0,
  "risk_underestimated_count": 35,
  "fusion_policy_match_count": 46,
  "request_more_info_count": 24,
  "final_channel_found_count": 49
}
```

## M6 SFT Dataset

M6 expands `data/router_sft_001.jsonl` from 10 examples to 150 schema-valid SFT examples. The purpose is router/verifier behavior stabilization before any fine-tuning, not knowledge injection.

The expansion is based on M5 base eval failures recorded in:

- `eval_results/base_eval_001.json`
- `eval_results/base_predictions_001.jsonl`
- `docs/base_eval_failure_analysis.md`

The dataset emphasizes:

- mode classification examples for all fixed router modes
- risk calibration, especially high and critical engineering cases
- concrete `needed_tools` selection
- domain-specific `verification.checks`
- stricter control of `request_more_info`, limited to missing code, missing files/attachments, or missing comparison targets
- explicit fusion decisions for `managed_fusion` and `self_fusion_lite`

Fine-tuning has still not been run. Run schema validation before any future training:

```bash
python scripts/validate_router_json.py data/router_sft_001.jsonl --mode sft-jsonl
```

## M6.5 Pre-LoRA Audit

M6.5 adds a pre-LoRA audit step for the expanded SFT dataset. It does not load the model and does not start training.

Run the audit with:

```bash
python scripts/audit_router_sft.py data/router_sft_001.jsonl evals/router_eval_001.jsonl
```

The audit reports:

- total rows
- mode, risk, and focus distributions
- empty or generic-only `verification.checks`
- empty `needed_tools`
- `request_more_info` and `local_rag` counts
- schema-valid row count
- exact and near-duplicate prompt overlap between SFT and eval data

M6.5 audit result:

```text
total_rows: 150
schema_valid_count: 150
schema_invalid_count: 0
parse_error_count: 0
request_more_info_count: 16
local_rag_count: 3
needed_tools_empty_count: 27
verification_empty_count: 0
verification_generic_only_count: 0
exact_match_count: 0
similar_candidate_count: 0
```

Environment snapshots for the `gptoss20b` conda environment are stored in:

- `requirements-gptoss20b-freeze.txt`
- `environment-conda-list-gptoss20b.txt`

`scripts/train_router_lora.py` supports `--max-samples` for future human-approved small training runs, but training still requires explicit `--run-train`. Do not run `--run-train` until a human approves the next milestone.

## M7 LoRA Dry-Run Finding

M7 attempted a 2-sample LoRA dry-run with `openai/gpt-oss-20b` on RTX5080 through the standard Transformers/TRL/PEFT path.

Result summary:

- `openai/gpt-oss-20b` MXFP4 inference and model loading worked locally.
- The dataset preview path worked without model loading or training.
- `q_proj` target modules were detected during the attempted dry-run.
- Standard Transformers/PEFT LoRA training on MXFP4 stopped before training because MXFP4 training is not supported by that path.
- No OOM occurred before the stop.
- No adapter was saved.
- The 10-sample dry-run was not executed.

The current operating decision is to treat `openai/gpt-oss-20b` as an inference-only router/verifier on RTX5080. `scripts/train_router_lora.py` now blocks `--run-train --load-strategy mxfp4-auto` before model loading to prevent repeating this unsupported route.

Detailed findings are in `docs/m7_lora_dryrun_findings.md`. Adapter directories under `adapters/` and operational logs under `logs/*.md` are not managed by git.

## M7b Prompt Router V2

M7b improves the inference-only base router without fine-tuning. It uses a stricter system prompt, five compact few-shot examples, and one schema repair attempt only when JSON parsing or schema validation fails.

Prompt files:

- `prompts/router_system_v2.md`
- `prompts/router_fewshot_v2.jsonl`

Run the full prompt-router v2 eval with:

```bash
conda run -n gptoss20b python scripts/eval_router.py \
  --eval-file evals/router_eval_001.jsonl \
  --schema schemas/router_output.schema.json \
  --model-name openai/gpt-oss-20b \
  --local-files-only \
  --trust-remote-code \
  --system-prompt-file prompts/router_system_v2.md \
  --fewshot-file prompts/router_fewshot_v2.jsonl \
  --prompt-version prompt-router-v2 \
  --repair-json-once \
  --max-new-tokens 256 \
  --temperature 0 \
  --results eval_results/prompt_router_v2_eval_001.json \
  --predictions-out eval_results/prompt_router_v2_predictions_001.jsonl \
  --markdown-log logs/prompt_router_v2_eval_001.md
```

Saved result files:

- `eval_results/prompt_router_v2_smoke_001.json`
- `eval_results/prompt_router_v2_eval_001.json`
- `eval_results/prompt_router_v2_predictions_001.jsonl`
- `docs/prompt_router_v2_eval_report.md`

M7b result summary versus M5 base eval:

```text
schema_valid_count: 44 -> 50
expected_mode_match_count: 16 -> 44
must_tools_contained_count: 9 -> 42
must_verification_contained_count: 0 -> 7
risk_underestimated_count: 35 -> 16
request_more_info_count: 24 -> 9
repair_attempted_count: 0 -> 1
repair_success_count: 0 -> 1
```

`openai/gpt-oss-20b` remains inference-only on RTX5080. No fine-tuning, LoRA dry-run, or adapter training was run for M7b.

## M8 Router Model Candidate Survey

M8 starts the survey of router model candidates beyond `openai/gpt-oss-20b`.
The candidate comparison is in `docs/router_model_candidates.md`.

The first experiment target is `Qwen/Qwen3.5-4B`, pending a separate approval
step for any model download, model load, baseline eval, or LoRA dry-run.
`openai/gpt-oss-20b` remains the inference-only router/verifier baseline on
RTX5080.

## Environment Success Memo

Current local environment status:

- Conda environment: `gptoss20b`
- Python: `3.12.13`
- torch: `2.11.0+cu128`
- transformers: `5.12.1`
- accelerate: `1.14.0`
- datasets: `5.0.0`
- peft: `0.19.1`
- trl: `1.6.0`
- bitsandbytes: `0.49.2`
- triton: `3.6.0`
- kernels: `0.14.1`
- `fsspec` should be pinned to `2026.4.0`
- Use `python -m pip`, not bare `pip`
- `openai/gpt-oss-20b` base inference succeeded
- Final-only JSON extraction succeeded
- `logs/*.md` are operational logs and are not managed by git
- Fine-tuning has not been run yet
