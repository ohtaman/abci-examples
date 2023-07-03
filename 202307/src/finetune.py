import datetime
import json
import logging
import os
import pathlib
import sys
from itertools import chain

import datasets
import fire
import peft
import torch
import yaml

import transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_dataset(train_file: str, valid_size: float=0.1, seed: int=42):
    dataset = datasets.load_dataset(
        'json', data_files=train_file, split='train'
    )
    return dataset.train_test_split(valid_size, shuffle=True, seed=seed)


def group_texts(examples, block_size: int):
    concatenated_examples = {
        k: list(
            chain(
                *
                examples[k])) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess(
        examples,
        tokenizer,
        input_template: str,
        block_size: int = 512,
        num_proc: int=4):
    prompts = examples.map(
        lambda x: {
            'text': input_template.format_map(x)
        }
    )
    tokenized = prompts.map(
        lambda x: tokenizer(
            x['text']),
        batched=True,
        num_proc=num_proc,
        remove_columns=prompts['train'].features)
    grouped = tokenized.map(
        lambda x: group_texts(
            x,
            block_size),
        batched=True,
        num_proc=num_proc)
    return grouped


def main(model_name: str, config_file: str):
    # 設定ファイルの読み込み
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
    
    # config['model'] は AutoModelForCausalLM.from_pretrained で読み込む際のパラメータ
    # モデル名を設定ファイルではなく cli の引数として持てるようにしているので、ここで config['model'] に設定
    config['model']['pretrained_model_name_or_path'] = model_name

    # 出力先ディレクトリの設定
    output_dir = pathlib.Path(os.path.expandvars(config['outputs']['dirname']))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 出力先ディレクトリの中に、モデル名のディレクトリを作成し、訓練結果の保存先とする
    model_output_dir = output_dir.joinpath(model_name)
    config['training']['output_dir'] = model_output_dir

    # 出力先ディレクトリに、最終的な設定値を保存しておく
    with open(output_dir.joinpath('config.yaml'), 'w') as o_:
        yaml.dump(config, o_)


    # データセット, トークナイザ, モデルのロード
    logger.info(f'load datasets')
    dataset = load_dataset(**config['data'])


    logger.info(f"load tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f'set pad_token to {tokenizer.pad_token}')


    logger.info(f"load model: {config['model']}")
    model = transformers.AutoModelForCausalLM.from_pretrained(**config["model"], torch_dtype=torch.float16)

    # 訓練対象の重みの指定
    # config['finetuning']['trainables'] に設定されていない重みについては requires_grad = False として訓練対象から除外する
    if 'trainables' in config.get('finetuning', {}):
        trainable_params = config['finetuning']['trainables']
        for name, param in model.named_parameters():
            trainable = False
            for trainable_param in trainable_params:
                if trainable_param in name:
                    trainable = True
            if not trainable:
                param.requires_grad = False
    # 訓練対象の重みについては、訓練の安定性の観点から float32 にしておく
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                print('convert to float32')
            print(f'tune {name} with {param.numel()} params')


    # モデルによっては以下のエラーが出るので暫定的な対応
    # AttributeError: 'function' object has no attribute '__func__'. Did you mean: '__doc__'?
    if not hasattr(model.forward, '__func__'):
        model.forward.__func__ = model.__class__.forward


    # データセットの前処理. 以下の形式に変換する
    # {'input_ids': [token1, token2, ...]}
    lm_dataset = preprocess(dataset, tokenizer, config['input_template'])
    # data_collator では、訓練用にデータを加工する
    # DataCollatorForLanguageModeling では、'input_ids' を 'labels' に設定する
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = transformers.TrainingArguments(**config["training"])
    # warning が出るので、 use_cache = False としておく
    model.config.use_cache = False
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test'],
        args=training_args,
        data_collator=data_collator,
    )

    with torch.autocast('cuda'):
        result = trainer.train()
    model.save_pretrained(model_output_dir)


if __name__ == "__main__":
    fire.Fire(main)

