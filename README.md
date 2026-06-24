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

SFT データは `messages` 形式の JSONL です。assistant の内容は JSON 文字列のみで、chain-of-thought は含めません。

```json
{"id":"...","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{\"mode\":\"verify\",\"tools\":[\"python\"],\"risk\":\"medium\",\"needs_human_approval\":false,\"next_action\":\"...\"}"}]}
```

評価データは期待値を持つ JSONL です。

```json
{"id":"...","prompt":"...","expected_mode":"verify","must_tools":["python"],"min_risk":"medium"}
```

router 出力の基本フィールド:

```json
{
  "mode": "route | verify | ask_clarification",
  "tools": ["python"],
  "risk": "low | medium | high | critical",
  "needs_human_approval": false,
  "next_action": "short action",
  "checks": ["short check"],
  "notes": "short note"
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

Existing `data/router_sft_001.jsonl` and `evals/router_eval_001.jsonl` are preserved as initial M1/M2 samples and are not rewritten by M3.

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
