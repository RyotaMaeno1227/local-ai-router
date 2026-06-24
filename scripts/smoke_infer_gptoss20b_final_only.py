import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai/gpt-oss-20b"

def extract_final_message(text: str) -> str:
    """
    Extract final-channel assistant message from gpt-oss Harmony-style output.
    Falls back to the full generated text if the final channel is not found.
    """
    patterns = [
        r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
    return text.strip()

def main():
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a local scientific AI router specialized in FEM, "
                "computational mechanics, numerical methods, and scientific coding. "
                "Return only valid JSON in the final answer."
            ),
        },
        {
            "role": "user",
            "content": "このC言語のFEM剛性行列組み立てコードを確認してください。",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode full output for debugging.
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Decode generated portion only.
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    final_text = extract_final_message(generated_text)

    print("\n===== GENERATED RAW =====")
    print(generated_text)

    print("\n===== FINAL ONLY =====")
    print(final_text)

    print("\n===== JSON CHECK =====")
    try:
        parsed = json.loads(final_text)
        print("valid_json: true")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as e:
        print("valid_json: false")
        print(repr(e))

if __name__ == "__main__":
    main()
