from itertools import chain
import logging
import os
import pathlib
import pydoc

import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import datasets
import fire
import peft
import torch
import yaml

import transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_dataset(train_file: str, valid_size: float = 0.1, seed: int = 42):
    dataset = datasets.load_dataset("json", data_files=train_file, split="train")
    return dataset.train_test_split(valid_size, shuffle=True, seed=seed)


def group_texts(examples, block_size: int):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess(
    examples, tokenizer, prompt_template: str, block_size: int = 1024, num_proc: int = 4
):
    prompts = examples.map(
        lambda x: {"text": prompt_template.format_map(x)}
        # lambda x: {"text": prompt_template.format_map(x) + tokenizer.eos_token}
    )
    tokenized = prompts.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", max_length=block_size, truncation=True
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=prompts["train"].features,
    )
    return tokenized


def main(config_file: str, model_name: str = None, local_rank: int = None):
    # 分散処理用の初期設定
    deepspeed.init_distributed()

    # 設定ファイルの読み込み
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    # config['model'] は AutoModelForCausalLM.from_pretrained で読み込む際のパラメータ
    # モデル名を設定ファイルではなく cli の引数として持てるようにしているので、ここで config['model'] に設定
    if model_name is not None:
        config["model"]["pretrained_model_name_or_path"] = model_name

    # 出力先ディレクトリの設定
    output_dir = pathlib.Path(os.path.expandvars(config["outputs"]["dirname"]))
    logger.info(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 出力先ディレクトリの中に、モデル名のディレクトリを作成し、訓練結果の保存先とする
    model_output_dir = output_dir.joinpath(model_name)
    config["training"]["output_dir"] = model_output_dir.absolute()

    # 訓練関連のパラメータ
    training_args = transformers.TrainingArguments(**config["training"])
    if training_args.should_log:
        logger.info(training_args)

    # 出力先ディレクトリに、最終的な設定値を保存しておく
    # ファイルに保存するのはメインプロセスのみ
    if training_args.should_log:
        with open(output_dir.joinpath("config.yaml"), "w") as o_:
            yaml.dump(config, o_)

    # データセットのロード
    logger.info(f"load datasets")
    dataset = datasets.load_dataset(**config["dataset"])

    # トークナイザのロード
    logger.info(f"load tokenizer")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding=True, padding_side='left')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"set pad_token to {tokenizer.pad_token}")

    logger.info(f"load model: {config['model']}")
    # torch_dtypeを文字列から型に変換しておく
    if "torch_dtype" in config["model"]:
        config["model"]["torch_dtype"] = pydoc.locate(config["model"]["torch_dtype"])

    model = transformers.AutoModelForCausalLM.from_pretrained(**config["model"])
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    # load_in_8bit/load_in_4bit の場合は必要. V100 では load_in_8bit/load_in_4bit 正常に動作しないのでコメントアウト
    # model = peft.utils.prepare_model_for_kbit_training(model)

    # ベースモデルからLoRAモデルを生成
    logger.info(f"build peft model")
    lora_config = peft.LoraConfig(**config["lora"])
    peft_model = peft.get_peft_model(model, peft_config=lora_config)
    peft_model.print_trainable_parameters()

    # モデルによっては以下のエラーが出るので暫定的な対応
    # AttributeError: 'function' object has no attribute '__func__'. Did you mean: '__doc__'?
    if not hasattr(peft_model.forward, "__func__"):
        logger.info("add peft_model.forward.__func__")
        peft_model.forward.__func__ = peft_model.__class__.forward

    with training_args.main_process_first():
        # データセットの前処理. 以下の形式に変換する
        # {'input_ids': [token1, token2, ...]}
        logger.info("convert datasets with prompt_template")
        lm_dataset = preprocess(dataset, tokenizer, config["prompt_template"])
        if "test" not in lm_dataset:
            lm_dataset = lm_dataset["train"].train_test_split(test_size=1000)
        train_dataset = lm_dataset["train"]
        eval_dataset = lm_dataset["test"]

    # data_collator では、訓練用にデータを加工する
    # DataCollatorForLanguageModeling では、'input_ids' を 'labels' に設定する
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    callbacks = []
    if "early_stopping_callback" in config:
        callbacks.append(
            transformers.EarlyStoppingCallback(**config["early_stopping_callback"])
        )
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "loss"

    # warning が出るので、 use_cache = False としておく
    peft_model.config.use_cache = False
    trainer = transformers.Trainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    with torch.autocast("cuda"):
        # モデルの出力先にチェックポイントファイルがある場合は、レジュームする
        if list(model_output_dir.glob("checkpoint-*")):
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = False
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save peft model ralated files.
    # this can not save the states when zero3 enabled.
    peft_model.save_pretrained(model_output_dir)

    logger.info(f"saving latest checkpoint")
    zero3_checkpoint_dir = model_output_dir.joinpath("zero3")
    trainer.model_wrapped.save_checkpoint(zero3_checkpoint_dir)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        zero3_checkpoint_dir
    )  # already on cpu
    if training_args.should_save:
        peft_state_dict = peft.get_peft_model_state_dict(
            peft_model, state_dict=state_dict
        )
        torch.save(peft_state_dict, model_output_dir.joinpath("adapter_model.bin"))

    logger.info("successfully finished finetuning.")


if __name__ == "__main__":
    fire.Fire(main)
