import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "cyberagent/open-calm-7b"


def generate_text(model, tokenizer, prompt: str, max_new_tokens=128, **kwargs) -> str:
    # 文字列をトークンの列に変換
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # トークンの列を文字列に変換
    return tokenizer.decode(outputs.sequences[0])


def main():
    # トークナイザの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 文章生成
    print(generate_text(model, tokenizer, "日本で一番高い山は"))


if __name__ == "__main__":
    main()
