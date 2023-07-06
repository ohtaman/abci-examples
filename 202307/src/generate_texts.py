import datetime
import json
import logging
import os
import pathlib
import sys
import pydoc

import datasets
import fire
import torch
import yaml

import transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_dataset(test_file: str):
    return datasets.load_dataset("json", data_files=test_file, split="train")


def generate_text_fn(model, tokenizer, args: dict = {}):
    def generate_text(prompt: str) -> str:
        input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = input_tokens.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **args,
            )
            output_tokens = outputs.sequences[0, input_length:-1]

        return tokenizer.decode(output_tokens)

    return generate_text


def main(config_file: str, model_name: str = None):
    # 設定ファイルの読み込み
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    # config['model'] は AutoModelForCausalLM.from_pretrained で読み込む際のパラメータ
    # モデル名を設定ファイルではなく cli の引数として持てるようにしているので、ここで config['model'] に設定
    if model_name is not None:
        config["model"]["pretrained_model_name_or_path"] = model_name

    # 出力先ディレクトリの設定
    output_dir = pathlib.Path(os.path.expandvars(config["outputs"]["dirname"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力先ディレクトリに、最終的な設定値を保存しておく
    with open(output_dir.joinpath("config.yaml"), "w") as o_:
        yaml.dump(config, o_)

    # データセットのロード
    logger.info(f"load datasets")
    dataset = load_dataset(test_file=config["data"]["test_file"])
    dataset = dataset.shuffle(seed=42)

    # トークナイザのロード
    logger.info(f"load tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"set pad_token to {tokenizer.pad_token}")

    # モデルのロード
    logger.info(f"load model: {config['model']}")
    # torch_dtypeを文字列から型に変換しておく
    if "torch_dtype" in config["model"]:
        config["model"]["torch_dtype"] = pydoc.locate(config["model"]["torch_dtype"])
    model = transformers.AutoModelForCausalLM.from_pretrained(**config["model"])

    generate_text = generate_text_fn(model, tokenizer, config["generate"])
    output_file = output_dir.joinpath(config["outputs"]["filename"])
    with open(output_file, "w") as o_:
        for i, data in zip(range(config["data"]["n_examples"]), dataset):
            if i % 10 == 0:
                logger.info(f"processing {i}th data.")
            prompt = config["prompt"].format_map(data)
            generated = generate_text(prompt)
            json.dump(dict(prompt=prompt, complete=generated), o_, ensure_ascii=False)
            o_.write("\n")
            o_.flush()


if __name__ == "__main__":
    fire.Fire(main)
