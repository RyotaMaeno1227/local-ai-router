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
  docs/router/README.md
  adapters/
  logs/
```

Use the [Codex job completion template](docs/router/codex_job_completion_template.md)
to report commit state, documentation, validation, prohibited actions, and the
next approval-gated step at the end of each job.

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
- `docs/router/base_eval_failure_analysis.md`

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

Detailed findings are in `docs/router/m7_lora_dryrun_findings.md`. Adapter directories under `adapters/` and operational logs under `logs/*.md` are not managed by git.

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
- `docs/router/prompt_router_v2_eval_report.md`

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
The candidate comparison is in `docs/router/router_model_candidates.md`.

The first experiment target is `Qwen/Qwen3.5-4B`, pending a separate approval
step for any model download, model load, baseline eval, or LoRA dry-run.
`openai/gpt-oss-20b` remains the inference-only router/verifier baseline on
RTX5080.

## M9 Qwen3.5 4B Router Eval

M9 evaluated `Qwen/Qwen3.5-4B` as a base router model using the multimodal
Transformers path: `AutoProcessor` + `AutoModelForMultimodalLM`.
The run used the prompt-router v2 system prompt, five few-shot examples, and one
schema repair attempt. Fine-tuning, LoRA dry-run, and adapter training were not
run.

Saved result files:

- `eval_results/qwen35_4b_eval_smoke_001.json`
- `eval_results/qwen35_4b_eval_001.json`
- `eval_results/qwen35_4b_predictions_001.jsonl`
- `docs/router/qwen35_4b_router_eval_report.md`

M9 comparison against `openai/gpt-oss-20b` prompt-router v2:

```text
schema_valid_count: 50 -> 50
expected_mode_match_count: 44 -> 45
must_tools_contained_count: 42 -> 37
must_verification_contained_count: 7 -> 7
risk_underestimated_count: 16 -> 3
request_more_info_count: 9 -> 8
repair_attempted_count: 1 -> 0
repair_success_count: 1 -> 0
```

Qwen3.5 4B is viable as a base router candidate, with lower risk
underestimation but weaker `needed_tools` containment than GPT-OSS
prompt-router v2. Any LoRA/QLoRA work still requires separate approval.

## M10 Qwen3.5 4B LoRA Dry-Run

M10 ran only a LoRA dry-run for `Qwen/Qwen3.5-4B`. Full fine-tuning and
150-row training were not run.

`scripts/train_router_lora.py` now supports the Qwen multimodal text-only path:

- `--model-family multimodal-lm`
- `--processor-name`
- `--torch-dtype`
- `--list-lora-targets`

The target module check found:

```text
q_proj: 8
v_proj: 8
k_proj: 8
o_proj: 8
gate_proj: 32
up_proj: 32
down_proj: 32
```

Dry-run results:

```text
2 samples:  q_proj, r=2, trainable params 172,032, train_loss 2.987, adapter saved
10 samples: q_proj/v_proj, r=4, trainable params 458,752, train_loss 2.771, adapter saved
```

No CUDA OOM occurred, no NaN loss was observed, and adapters were saved under
`adapters/dryrun-qwen35-4b-router-lora-*`. Adapter artifacts and `logs/*.md`
remain git-ignored.

M9 failure analysis for future approved training is in
`docs/router/qwen35_4b_eval_failure_analysis.md`. The main priorities are canonical
verification check names, richer `needed_tools`, research novelty routing, and
conservative nonlinear/contact risk calibration.

## M11a Qwen3.5 4B Router LoRA v001-small

M11a completed the first approved small LoRA training run for
`Qwen/Qwen3.5-4B`. It used `data/router_sft_train_050.jsonl`, a fixed 50-row
subset emphasizing verification, tool selection, request-more-info boundaries,
and mode/risk coverage. The prohibited 150-row full training run was not run.

Training completed for one epoch with `q_proj`/`v_proj`, `r=4`, `alpha=8`,
maximum length 256, batch size 1, and gradient accumulation 4. The final
training loss was 2.746; no CUDA OOM or non-finite loss occurred. The adapter
is stored under `adapters/qwen35-4b-router-lora-v001-small` and is not managed
by git.

On the existing 50-case eval, LoRA improved `must_tools_contained_count` from
37 to 39. Mode match (45), verification containment (7), and risk
underestimation (3) were unchanged. The independent 30-case holdout produced
30/30 schema-valid outputs, 28 mode matches, 22 tool matches, 0 exact
verification matches, and 8 risk underestimates. This small adapter is not yet
evidence for broad generalization.

Results:

- `eval_results/qwen35_lora_v001_small_eval_001.json`
- `eval_results/qwen35_lora_v001_small_predictions_001.jsonl`
- `eval_results/qwen35_lora_v001_small_holdout_eval_001.json`
- `eval_results/qwen35_lora_v001_small_holdout_predictions_001.jsonl`
- `docs/router/qwen35_lora_v001_small_eval_report.md`

Operational logs remain under `logs/` and are not managed by git.

## M11b Base Holdout and Verification Vocabulary

M11b evaluated the unadapted `Qwen/Qwen3.5-4B` on the 30-case holdout under
the same prompt and generation settings as LoRA v001-small. No training or
adapter update was run.

Base scored 27 mode matches, 22 tool matches, 0 strict verification matches,
and 8 risk underestimates. LoRA scored 28, 22, 0, and 8 respectively. The LoRA
improvement is limited to one mode boundary; broader holdout improvement was
not demonstrated.

Verification analysis found a vocabulary contract mismatch: predictions use
natural-language labels while holdout expectations use canonical snake_case
identifiers. The existing few-shot prompt also teaches natural-language
labels. Canonical values and migration aliases are documented in
`docs/router/router_canonical_vocabulary.md`; detailed comparison is in
`docs/router/qwen35_holdout_and_verification_analysis.md`.

M11b artifacts:

- `eval_results/qwen35_4b_holdout_eval_001.json`
- `eval_results/qwen35_4b_holdout_predictions_001.jsonl`
- `scripts/analyze_verification_labels.py`
- `docs/router/router_canonical_vocabulary.md`
- `docs/router/qwen35_holdout_and_verification_analysis.md`

## M12 Canonical Vocabulary Repair and Prompt-Router v3

M12 introduced prompt-router v3 and canonicalized the verification vocabulary
contract without additional training. `verification.checks` now uses only
lowercase snake_case identifiers; explanatory text remains in
`verification.reason`. The non-destructive normalizer produced:

- `data/router_sft_002_canonical.jsonl` (150 rows)
- `data/router_sft_train_050_canonical.jsonl` (50 rows)

Both source SFT datasets were already canonical: zero labels changed and zero
unknown labels were found. The prompt, especially its v2 few-shot examples,
was the source of the vocabulary mismatch.

On the canonical holdout, base prompt v3 improved verification from 0 to 8,
tools from 22 to 24, and mode matches from 27 to 28. LoRA v001-small prompt v3
improved verification from 0 to 8 and tools from 22 to 26. Risk
underestimation remained 8 for both. The original 50-case eval still uses
legacy natural-language expectations, so its v3 strict-verification score is
not comparable to v2.

Qwen3.5 4B remains the local router candidate for RTX5080 16GB. Qwen3.5 9B is
not in scope. No fine-tuning, LoRA dry-run, adapter update, or 150-row training
was run in M12.

M12 artifacts:

- `prompts/router_system_v3.md`
- `prompts/router_fewshot_v3.jsonl`
- `scripts/normalize_router_vocabulary.py`
- `docs/router/router_vocabulary_repair_report.md`
- `docs/router/qwen35_prompt_v3_eval_report.md`
- `eval_results/qwen35_4b_prompt_v3_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_holdout_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_holdout_eval_001.json`

## M13 Canonical Eval and Vocabulary Whitelist

M13 migrated the original 50-case eval to
`evals/router_eval_001_canonical.jsonl` without modifying the legacy source.
All 86 legacy verification labels were explicitly mapped to 89 canonical
occurrences with zero unresolved labels. Legacy v2 and canonical v3
verification scores are not directly interchangeable.

`scripts/validate_router_vocab.py` now checks prediction, SFT, and eval JSONL
for allowed mode, risk, tool, verification, and fusion vocabulary. Strict mode
returns nonzero for unknown values. Prediction audits are non-destructive and
found a small number of field-placement or invented-label errors, documented
in `docs/router/qwen35_prompt_v3_canonical_eval_report.md`.

On canonical 50 cases, Qwen3.5 base scored 45 mode, 39 tool, and 17
verification matches with 3 risk underestimates. LoRA v001-small scored 45,
39, and 16 with 4 risk underestimates. The adapter did not improve the
canonical 50-case result, though it retains a two-case tool advantage on the
canonical holdout.

Qwen3.5 4B with prompt-router v3 remains the supported local candidate on
RTX5080 16GB. Qwen3.5 9B is not in scope. No additional training, LoRA
dry-run, adapter update, API access, or 150-row training occurred in M13.

M13 artifacts:

- `evals/router_eval_001_canonical.jsonl`
- `scripts/canonicalize_router_eval.py`
- `scripts/validate_router_vocab.py`
- `docs/router/router_eval_canonicalization_report.md`
- `docs/router/qwen35_prompt_v3_canonical_eval_report.md`
- `eval_results/qwen35_4b_prompt_v3_canonical_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_eval_001.json`

## M14 Inference-Time Vocabulary Repair

M14 adds canonical vocabulary validation to `scripts/eval_router.py` after
JSON parsing and schema validation. If a schema-valid output contains unknown
`needed_tools` or `verification.checks`, the model receives one JSON-only
repair request. The repaired output is then revalidated against both schema
and vocabulary. `--vocab-strict` exits nonzero after saving results if any
case remains invalid.

The base canonical 50 run repaired 3/3 invalid cases and finished with 50/50
vocab-valid outputs. The base holdout repaired 2/2 and finished at 30/30. The
LoRA comparison repaired 3/3 and finished at 50/50. Unknown tool and
verification label counts were zero in all final M14 predictions.

Router quality metrics did not change from M13: base canonical remained at 45
mode, 39 tool, and 17 verification matches with 3 risk underestimates. Base
holdout remained at 28, 24, and 8 with 8 risk underestimates. Vocabulary retry
enforces the output contract but does not by itself improve routing quality.

Qwen3.5 4B base with prompt-router v3 remains the standard local router
candidate. LoRA v001-small remains comparison-only and is not the selected
adapter. Qwen3.5 9B is out of scope. No additional training, LoRA dry-run,
adapter update, API access, or 150-row training occurred in M14.

M14 artifacts:

- `scripts/router_vocab.py`
- `docs/router/qwen35_vocab_repair_eval_report.md`
- `eval_results/qwen35_4b_prompt_v3_canonical_vocab_repair_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_holdout_vocab_repair_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_vocab_repair_eval_001.json`

## M15 Risk and Verification Gap Analysis

M14 stabilized schema and canonical vocabulary output. M15 analyzes the
remaining quality gaps without loading a model or running additional training.
The analysis covers risk underestimation, exact verification completeness,
required tools, mode selection, and `request_more_info` boundaries across the
canonical 50 and holdout 30 results.

The main gaps are conservative risk calibration for nonlinear/contact/code
tasks and incomplete verification sets. Across both evals,
`assumption_check` was missing 15 times and `python` was missing 10 times. The
analysis also identified a substring-scoring edge case, so future canonical
verification evaluation should use exact label membership.

`docs/router/router_sft_v002_design.md` proposes a reviewed 90-row canonical candidate
set emphasizing risk, verification completeness, tools, and clarification
contrast pairs. This remains a design stage: no v002 JSONL, fine-tuning, LoRA
dry-run, adapter update, or 150-row training was performed.

M15 artifacts:

- `scripts/analyze_router_quality_gaps.py`
- `docs/router/qwen35_risk_verification_gap_analysis.md`
- `docs/router/router_sft_v002_design.md`

## M16 Router SFT v002 Candidate Data

M16 turns the M15 design into a 90-row canonical training candidate at
`data/router_sft_v002_candidate.jsonl`. The allocation is 15 contact, 15
nonlinear FEM, 15 code review, 15 paper/novelty, 10 FEM fundamentals, 10
RAG/API/fusion, and 10 `request_more_info` boundary rows. All 90 rows belong
to controlled pair or triplet groups that contrast evidence, artifact, risk,
or routing conditions.

The candidate passes router schema and strict vocabulary validation for all
90 rows. The audit found no exact prompt matches against either eval dataset
or either prior SFT dataset. Three near-match candidates are retained in the
audit report for human review rather than silently removed.

This is candidate data only. No fine-tuning, LoRA dry-run, adapter update, or
150-row full training was run in M16. Training requires a separate approval
after review of the candidate data and near-match report.

M16 artifacts:

- `data/router_sft_v002_candidate.jsonl`
- `scripts/build_router_sft_v002_candidate.py`
- `scripts/audit_router_sft_v002.py`
- `docs/router/router_sft_v002_candidate_audit.md`

## M16b Router SFT v002 Candidate Freeze

M16b reviewed and rewrote the three near-duplicate prompt candidates from the
M16 audit. The follow-up audit reports zero exact eval/SFT prompt matches and
zero near-match candidates at the configured 0.82 threshold, while retaining
all 90 rows, the area allocation, and controlled pair/triplet structure.

`data/router_sft_v002_candidate.jsonl` is now frozen as the candidate for the
next explicit training approval gate. Schema and strict vocabulary validation
remain 90/90. Freeze status does not authorize training: no fine-tuning, LoRA
dry-run, adapter update, model load, or 150-row training was performed.

M16b artifacts:

- `docs/router/router_sft_v002_freeze_note.md`
- `docs/router/router_sft_v002_candidate_audit.md`
- `data/router_sft_v002_candidate.jsonl`

## M17 Qwen3.5 4B LoRA v002-small

M17 completed the approved one-epoch Qwen3.5 4B LoRA v002-small run on the
90-row frozen candidate. The dataset SHA-256 was
`f451b8007cf9992ab4310625bd5d01078049e5bcb919d3ef6647b94873ec2332`.
Training completed 23 optimizer steps with aggregate `train_loss=2.637`, no
CUDA OOM, no non-finite loss, and a successful adapter save at
`adapters/qwen35-4b-router-lora-v002-small`.

On canonical 50, v002 scored 45 mode, 39 tools, 18 verification, and 4 risk
underestimates, with 49/50 vocabulary-valid outputs. Compared with base, this
is +1 verification but one additional risk underestimate and one failed
vocabulary repair. On holdout 30, v002 scored 28 mode, 26 tools, 8
verification, and 8 risk underestimates; only tool containment improved over
base.

The adoption decision is **keep base standard and retain v002 as reference**.
The standard remains Qwen3.5 4B base with prompt-router v3 and schema/vocab
repair. Adapter weights and operational logs are git-ignored. No 150-row
training, Qwen3.5 9B use, GPT-OSS fine-tuning, dependency reinstall, or API
connection occurred in M17.

M17 artifacts:

- `docs/router/qwen35_lora_v002_small_training_report.md`
- `docs/router/qwen35_lora_v002_small_eval_report.md`
- `eval_results/qwen35_lora_v002_small_prompt_v3_canonical_vocab_repair_eval_001.json`
- `eval_results/qwen35_lora_v002_small_prompt_v3_canonical_vocab_repair_predictions_001.jsonl`
- `eval_results/qwen35_lora_v002_small_prompt_v3_holdout_vocab_repair_eval_001.json`
- `eval_results/qwen35_lora_v002_small_prompt_v3_holdout_vocab_repair_predictions_001.jsonl`

## M18 Standard Router Decision

M18 closes the current Qwen3.5 4B LoRA comparison. The standard router is
`Qwen/Qwen3.5-4B` base with no adapter, prompt-router v3, one schema repair,
strict vocabulary validation, and one vocabulary repair. LoRA v001-small and
v002-small remain local reference artifacts and are not selected adapters.

The adapters produced narrow holdout tool gains but did not improve the main
risk and verification gaps across both evals. V001 regressed canonical
verification and risk. V002 gained one canonical verification match but
regressed risk and left one strict vocabulary failure. Additional small LoRA
runs are paused until a materially new training signal and independent eval
justify reopening them.

The next phase is the external API dispatcher, local RAG, citation and
overclaim guards, and deterministic FEM verification workflow. M18 only
documents this architecture: no training, model load, adapter update, API
connection, or package installation occurred.

M18 artifacts:

- `docs/router/qwen35_lora_postmortem_and_router_decision.md`
- `docs/router/router_standard_config.md`
- `docs/router/next_phase_external_api_rag_design.md`

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
- GPT-OSS fine-tuning has not been run; Qwen3.5 4B LoRA v001-small used only
  the approved 50-row subset for one epoch
