# Router Model Candidate Survey

M8 is a documentation-only survey. No model was downloaded, loaded, fine-tuned,
or called through an API during this milestone.

## Decision

The first experiment target is `Qwen/Qwen3.5-4B`.

Reasoning:

- It is the smallest listed candidate that is plausibly strong enough to act as
  a fine-tuned router hub.
- It is expected to fit RTX5080 16GB inference with more margin than
  `openai/gpt-oss-20b`.
- A 4B-class model is the most practical next target for a controlled
  LoRA/QLoRA dry-run after separate approval.

Before any future download or model load, verify the exact license, disk size,
Transformers support, tokenizer/chat template behavior, text-only loading path,
and PEFT target module names.

## Operating Constraints

- `openai/gpt-oss-20b` remains an inference-only router/verifier on RTX5080.
- M8 does not approve model downloads, model loads, API calls, LoRA dry-runs, or
  adapter training.
- Any future experiment must start with a new explicit approval step.
- Router SFT remains behavior stabilization only. It is not knowledge injection.
- Router outputs must remain schema-valid JSON without chain-of-thought.

## Candidate Matrix

| Candidate | Model ID | Role | Size | Modality | Expected RTX5080 inference feasibility | Expected RTX5080 fine-tuning feasibility | Likely training route | Risk | First experiment plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B | `Qwen/Qwen3.5-4B` | Primary fine-tuned router hub candidate | 4B language component; HF page also reports 5B total params | Vision-language model, usable as text router if text-only path is supported | High, pending local dependency check and text-only load behavior | Likely feasible for LoRA/QLoRA at short router lengths, batch size 1, low rank | PEFT/TRL LoRA or QLoRA after baseline eval; discover target modules after approved load | Newer architecture may require exact Transformers support; multimodal components can add memory and processor complexity | First approved experiment: source/license preflight, then 3-case baseline eval, then 2-sample LoRA dry-run only if approved |
| Gemma 4 E2B | `google/gemma-4-E2B` | Small fallback fine-tuned router candidate | E2B class | Multimodal: text and image, with audio on small models | Very high expected feasibility | Likely feasible with the most VRAM margin among listed local candidates | PEFT/TRL LoRA or QLoRA; keep router task text-only first | Router behavior quality may be weaker than 4B+ candidates; Gemma 4 multimodal processor path needs confirmation | Keep as fallback if Qwen3.5-4B is blocked by dependency, license, or memory issues |
| Gemma 4 E4B | `google/gemma-4-E4B` | Alternative local fine-tuned router candidate | E4B class | Multimodal: text and image, with audio on small models | High expected feasibility | Likely feasible, but less margin than E2B | PEFT/TRL LoRA or QLoRA; start with text-only router SFT | Same Gemma 4 processor/training uncertainty as E2B, with higher memory pressure | Evaluate only after Qwen3.5-4B or E2B decision, using the same 3-case baseline gate |
| GPT-OSS 20B | `openai/gpt-oss-20b` | Existing inference-only router/verifier baseline | 20B | Text generation | Already proven locally for base inference and prompt-router v2 eval | Not feasible through the standard Transformers/PEFT MXFP4 path on RTX5080; BF16 needs much larger VRAM | No local training route in this repo; use prompt-router v2, few-shot, schema repair, and verifier prompts | MXFP4 training is unsupported by the standard path; model can still under-select verification tools without prompting | Continue as inference-only baseline and verifier. Do not retry MXFP4 LoRA training |
| GLM-5.2 | `zai-org/GLM-5.2` / GLM-5.2 external service | External expert candidate for long-horizon engineering review | HF reports 753B params | Text generation | Not a practical RTX5080 16GB local candidate | Not a local fine-tuning candidate | No local training. Use only as a separately approved external expert route | API/provider use is outside current permission scope; local deployment is far beyond this workstation target | Document-only for now. If approved later, define an external-expert protocol and cost/privacy gate before any connection |

## First Experiment: Qwen/Qwen3.5-4B

The next approved experiment should be intentionally small:

1. Re-check the model card, license, expected download size, and dependency
   requirements.
2. Confirm whether a text-only inference path can avoid unnecessary multimodal
   memory overhead.
3. Run a 3-case baseline router eval with schema validation.
4. If baseline loading and eval are stable, run a 2-sample LoRA/QLoRA dry-run
   with short `max_length`, batch size 1, and low rank.
5. Stop immediately on OOM, unsupported target modules, adapter-save failure, or
   NaN loss.

This plan is not approval to run the experiment.

## Source Notes

Checked on 2026-06-25:

- Qwen model card: https://huggingface.co/Qwen/Qwen3.5-4B
- Gemma 4 E2B model card: https://huggingface.co/google/gemma-4-E2B
- Gemma 4 E4B model card: https://huggingface.co/google/gemma-4-E4B
- Gemma 4 Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/gemma4
- GPT-OSS 20B model card: https://huggingface.co/openai/gpt-oss-20b
- GLM-5.2 model card: https://huggingface.co/zai-org/GLM-5.2
- GLM-5.2 Z.AI docs: https://docs.z.ai/guides/llm/glm-5.2
