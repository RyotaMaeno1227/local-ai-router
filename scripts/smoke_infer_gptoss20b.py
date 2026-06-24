import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai/gpt-oss-20b"

def main():
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
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
                "Return only valid JSON."
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

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\n===== OUTPUT =====")
    print(text)

if __name__ == "__main__":
    main()
