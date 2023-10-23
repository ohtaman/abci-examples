# PCCC AI/機械学習技術部会 第5回ワークショップ「大規模言語モデルハンズオン」

2023/10/23 開催の[大規模言語モデルハンズオン](https://www.pccluster.org/ja/event/2023/09/231023-ai-ws.html) の資料です. [第1回大規模言語モデル分散学習ハッカソン](https://abci.ai/event/2023/06/13/ja_event.html) 用の[サンプルプログラム](../202307)を元に作成しています.


### 内容

1. ABCIの基本的な使い方
2. 仮想環境の構築
3. LLMを用いた文章生成
4. HuggingFace Transformers の Trainer を用いた訓練
5. PEFT を用いた訓練

## ABCI の基本的な使い方

ABCI（AI Bridging Cloud Infrastructure, AI橋渡しクラウド）は以下の構成となっています。インタラクティブノードにログインして、そこから計算ノードにジョブを投入することで、実際の計算を行います.

![system-overview](https://docs.abci.ai/ja/img/abci_system_ja.svg)
**https://docs.abci.ai/ja/system-overview/**
  

具体的には、以下のようなスクリプトファイルを用意し、ジョブを投入するコマンドである `qsub` の引数として指定します. ジョブのパラメータは、以下の例のようにファイルの冒頭に記述することも、 `qsub` の引数として指定することもできます.


```bash
#!/bin/bash
#$ -l rt_G.small=1
#$ -j y
#$ -N hello_abci
#$ -o logs/
#$ -cwd

echo "This is {$USER}'s first job on ABCI!"

```
**scripts/hello_abci.sh**

投入コマンドは以下の通りです（ここではグループ名を `qsub` の引数として渡しています. `$ ` で始まる行が入力、それ以外は出力を表しています）

```bash
$ export GROUP=<your group name>
$ qsub -g $GROUP scripts/hello_abci.sh
Your job 40115860 ("hello_abci") has been submitted
```

標準出力が `-o` で指定したファイルもしくはディレクトリに出力されます

```bash
$ cat logs/hello_abci.o40115860
This is <user_name>'s first job on ABCI!
```

投入したジョブは `qstat` で確認できます（上記の `hello_abci.sh` の場合は一瞬で処理が終了するため、確認できないかもしれません）


```bash
$ qstat
job-ID     prior   name       user         state submit/start at     queue                          jclass                         slots ja-task-ID 
------------------------------------------------------------------------------------------------------------------------------------------------
  40115866 0.00000 finetune_l xxxxxxx   qw    07/06/2023 04:42:08                                                                  40        
```

処理を中断したい時は `qdel <job_id>` を使います.


```bash
$ qdel 40115866
xxxxxxxx has registered the job 40115866 for deletion
```


詳細は[ユーザーガイド - ジョブ実行](https://docs.abci.ai/ja/job-execution/)をご覧ください

## 環境構築

### 仮想環境の構築と Environment Modules

ABCI では Environment Modules を利用することで、ユーザーごとに必要なライブラリのみをロードできるようになっています.

利用可能なモジュールは `module avail` で確認できます.

```bash
$ module avail
---------------------------------------------------------------------------------------------------------------------------------- /apps/modules/modulefiles/rocky8/compilers ----------------------------------------------------------------------------------------------------------------------------------
gcc/8.5.0  gcc/12.2.0  intel/2023.0.0  

---------------------------------------------------------------------------------------------------------------------------------- /apps/modules/modulefiles/rocky8/devtools -----------------------------------------------------------------------------------------------------------------------------------
cmake/3.26.1  go/1.20  intel-advisor/2023.0  intel-inspector/2023.0  intel-itac/2021.8.0  intel-mkl/2023.0.0  intel-vtune/2023.0.0  julia/1.8  openjdk/1.8.0.362  openjdk/11.0.18.0.10  openjdk/17.0.6.0.10  python/3.10/3.10.10  python/3.11/3.11.2  R/4.2.3  

------------------------------------------------------------------------------------------------------------------------------------ /apps/modules/modulefiles/rocky8/gpgpu ------------------------------------------------------------------------------------------------------------------------------------
cuda/10.2/10.2.89  cuda/11.2/11.2.2  cuda/11.5/11.5.2  cuda/11.8/11.8.0  cuda/12.1/12.1.1  cudnn/8.1/8.1.1  cudnn/8.4/8.4.1  cudnn/8.7/8.7.0  cudnn/8.9/8.9.2   nccl/2.6/2.6.4-1  nccl/2.9/2.9.9-1    nccl/2.12/2.12.12-1  nccl/2.15/2.15.5-1  nccl/2.18/2.18.1-1  
cuda/11.0/11.0.3   cuda/11.3/11.3.1  cuda/11.6/11.6.2  cuda/12.0/12.0.0  cudnn/7.6/7.6.5   cudnn/8.2/8.2.4  cudnn/8.5/8.5.0  cudnn/8.8/8.8.1  gdrcopy/2.3       nccl/2.7/2.7.8-1  nccl/2.10/2.10.3-1  nccl/2.13/2.13.4-1   nccl/2.16/2.16.2-1  
cuda/11.1/11.1.1   cuda/11.4/11.4.4  cuda/11.7/11.7.1  cuda/12.1/12.1.0  cudnn/8.0/8.0.5   cudnn/8.3/8.3.3  cudnn/8.6/8.6.0  cudnn/8.9/8.9.1  nccl/2.5/2.5.6-1  nccl/2.8/2.8.4-1  nccl/2.11/2.11.4-1  nccl/2.14/2.14.3-1   nccl/2.17/2.17.1-1  

------------------------------------------------------------------------------------------------------------------------------------- /apps/modules/modulefiles/rocky8/mpi -------------------------------------------------------------------------------------------------------------------------------------
hpcx-debug/2.12  hpcx-mt/2.12  hpcx-prof/2.12  hpcx/2.12  intel-mpi/2021.8  

------------------------------------------------------------------------------------------------------------------------------------ /apps/modules/modulefiles/rocky8/utils ------------------------------------------------------------------------------------------------------------------------------------
aws-cli/2.11  s3fs-fuse/1.91  singularitypro/3.9  
```

ロードしたいモジュールが見つかったら、 `module load <module name>` でロードします.

```bash
module load python/3.11 cuda/11.7 cudnn/8.6 hpcx/2.12
```

ロードしたモジュールは `module list` で確認できます.

```bash
$ module list
Currently Loaded Modulefiles:
 1) python/3.11/3.11.2   2) cuda/11.7/11.7.1   3) cudnn/8.6/8.6.0   4) hpcx/2.12
```

### Python の仮想環境構築

単一ノードでサンプルプログラムを実行するには、 python および cuda モジュールをロードした上で、以下の容量で仮想環境（venv）を構築します.
この作業はインタラクティブノードで実施可能です. また、一度実施しておけばよく、ジョブごとに実行する必要はありません.

```bash
$ git clone https://github.com/ohtaman/abci-examples.git
$ cd abci-examples/202307
$ module load python/3.11 cuda/11.7 cudnn/8.6 hpcx/2.12
$ python3 -mvenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

`pip freeze` で、必要なライブラリがインストールされていることを確認できます.

```bash
$ pip freeze
accelerate==0.23.0
aiohttp==3.8.6
aiosignal==1.3.1
anyio==4.0.0
appdirs==1.4.4
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
...
```

> [!NOTE]
> サンプルコードの修正なしに、Weights & Biases を使って訓練の経過を可視化することもできます.
> それには、 `wandb` パッケージのインストールと初期設定（ログイン）が必要となります.
> https://docs.wandb.ai/ja/quickstart


### bitsandbytes のインストール

bitsandbytes は言語モデルを 8bit/4bit に量子化し、GPUメモリ圧縮・高速化するライブラリです.
V100 では勾配をうまく取り扱えなくいことがあるようですが、推論では利用可能であり、また DeepSpeed を利用した際に若干高速化されるようです.

pip でもインストール可能ですが、更新が早いので GitHubから直接インストールすることをお勧めします. こちらもインタラクティブノードで実行可能です（仮想環境に入った状態で実行します）.

```bash
$ git clone https://github.com/timdettmers/bitsandbytes.git
$ cd bitsandbytes

$ CUDA_VERSION=117 make cuda11x_nomatmul
$ python setup.py install

$ cd ..
```

## 文章生成

今回のハンズオンでは、 [HuggingFace](https://huggingface.co/) に登録されている以下のモデルを使って、文章生成やファインチューニングを行います.

|モデル名|URL|備考|
|:--:|:--|:--|
| cyberagent/open-calm-7b |  https://huggingface.co/cyberagent/open-calm-7b |
| elyza/ELYZA-japanese-Llama-2-7b | https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b |
| cerebras/Cerebras-GPT-256M | https://huggingface.co/cerebras/Cerebras-GPT-256M |


HuggingFace に登録されているモデルで文章生成を行うには、`pipeline` を使う方法と、より細かい調整のできる `model.generate` メソッドを使う方法がありますが、ここでは `model.generate` メソッドを用います.


`src/generate_text.py` では、 `cyberagent/open-calm-7b` を使って「日本で一番高い山は」につづく文章を生成しています.

<details>
<summary>src/generate_text.py</summary>

```python
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

```

</details>

ABCIでこのコードを実行するには、`qrsh` を使って計算ノードにログインしてインタラクティブに実行するか、`qsub` でジョブを実行します.

```
$ qsub -g $GROUP scripts/generate_text.sh 
```

<details>
<summary>src/generate_text.py</summary>

```bash#!/bin/bash
#$ -l rt_G.large=1
#$ -j y
#$ -N generate_texts
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6
source .venv/bin/activate

# モデルやデータセットはデフォルトではホームディレクトリ以下のキャッシュフォルダにダウンロードされる
# ABCI ではホームディレクトリ以下の容量は限られているので、キャッシュディレクトリを scratch 領域に変更
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/

python src/generate_text.py
```
</details>

ジョブの結果は `logs/generate_texts.o<job id>` に保存されます

```bash
$ cat logs/generate_texts.o40812574 
Loading checkpoint shards: 100%|██████████| 2/2 [00:12<00:00,  6.36s/it]
日本で一番高い山は富士山ですが、日本では2番目に高い山になります。
富士山は標高3,776m、立山連峰は標高3,015m、そして、立山は日本百名山にも選ばれている、立山連峰の最高峰になります。
立山連峰は、立山、剱岳、白馬岳、そして、槍ヶ岳の4つの山の総称になります。
立山連峰は、立山、剱岳、白馬岳、そして、槍ヶ岳の4つの山の総称になります。立山連峰は、立山、剱岳、白馬岳、そして、槍ヶ岳の

```

「日本で一番高い山は」につづく文章が生成されているのがわかります.

## LLM とファインチューニング

ファインチューニングとは、事前学習済みの深層学習モデル（ベースモデルと呼びます）を、事前学習とは別のデータセットを使って再訓練することで、新しいタスクに適応させることを指します。タスクによっては新しい層を追加することもあります。

ファインチューニングを行うにあたって、どのパラメータを更新するかという自由度が生まれます. 今回は、以下の3パターンを取り扱います.

1. フルファインチューニング
   - 言語モデルのパラメータを全て訓練対象とします
2. は指定した層のファインチューニング
   - 訓練対象としたい層以外のパラメータを `requires_grad` を `False` とすることで、訓練対象からはずします
   - 訓練対象のパラメータが減るので、省メモリで訓練可能となります
   - 最終層をすげかえたモデルを構築し、すげかえた層のみを訓練することで、分類問題など事前訓練とは別のタスクに利用することも可能です
3. LoRA
   - ベースとなるモデルのパラメータに低ランク行列を加算するような構造のモデルを作成し、低ランク行列のみを訓練する手法です
   - 2022年に発表された手法で、非常に少ないパラメータ数で、フルファインチューニングと遜色ない精度を実現しています
   - 論文では、アテンション層のみを対象としていますが、より一般的には、任意の重み行列に対応可能です


### ファインチューニングに利用するデータセット

今回は [`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) を利用します. これは Databricks社の作成した [`databricks-dolly-15k`] を日本語に翻訳したもので、以下のような内容となっています.

- index: インデックス
- category: タスクの種類
- input: 入力データ（タスクの種類によっては無し）
- instruction: 指示内容
- output: instruction に対する、望ましい出力

<details>
<summary>kunishou/databricks-dolly-15k-ja のサンプルデータ</summary>

| input | category | output | instruction | index |
|:--|:--|:--|:--|--:|
| ヴァージン・オーストラリア航空（Virgin Australia Airlines Pty Ltd）はオーストラリアを拠点とするヴァージン・ブランドを冠する最大の船団規模を持つ航空会社です。2000年8月31日に、ヴァージン・ブルー空港として、2機の航空機、1つの空路を運行してサービスを開始しました。2001年9月のアンセット・オーストラリア空港の崩壊後、オーストラリアの国内市場で急速に地位を確立しました。その後はブリスベン、メルボルン、シドニーをハブとして、オーストラリア国内の32都市に直接乗り入れるまでに成長しました。| closed_qa | ヴァージン・オーストラリア航空は、2000年8月31日にヴァージン・ブルー航空として、2機の航空機で単一路線の運航を開始しました。 | ヴァージン・オーストラリア航空はいつから運航を開始したのですか？ | 0 |
| | classification | イコクエイラクブカ | 魚の種類はどっち？イコクエイラクブカとロープ  | 1 |
| | open_qa | ラクダは、長時間にわたってエネルギーと水分で満たされた状態を保つために、腰の脂肪を利用しています。| ラクダはなぜ水なしで長く生きられるのか？ | 2 |

</details>

LLMの入力は文字列である必要があるため、各行を文字列に変換する必要があります.
サンプルコードでは、設定ファイル（`config/` 以下のファイル）で変換のルールを決めています.

```yaml
prompt_template: |-
  ### Instruction:
  {instruction}
  ### Input:
  {input}
  ### Response:
  {output}
```

#### フルファインチューニングの例

ファインチューニングのコードは `src/finetune.py` にあります. このコードは、モデル名と設定ファイル名を指定するようになっています. ジョブ投入スクリプトは `scripts` 以下にあります.

フルファインチューニングには、推論と比較して非常に多くのメモリが必要となるため、ここでは小さなモデル `cerebras/Cerebras-GPT-256M` を指定しています.

設定ファイルは `config/config_finetune_full.yaml` にあります. パラメーター（`learning_rate`` など）を自由に動かして実行してみてください.

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune_full.yaml scripts/finetune.sh
```

訓練されたモデルは、設定ファイルの `outputs.dirname` で指定したディレクトリ以下にモデル名つきで保存されます.

また、訓練したモデルを使って文章を生成することもできます. それには、前述の `scr/generate_text.py` において、MODEL_NAME をモデルの保存先ディレクトリ名に修正します（user_name や job_id は適宜修正してください）.


```python
# MODEL_NAME = "cyberagent/open-calm-7b"
MODEL_NAME = "/scratch/<user_name>/models/<job_id>/cerebras/Cerebras-GPT-256M"
```

#### 更新対象の層を指定したファインチューニングの例

サンプルプログラムでは、設定ファイルで `finetuning.trainables` という項目を設定することで、訓練対象を指定できるようになっています. 指定できる層の名称は、モデルのアーキテクチャ毎に異なります.

設定ファイルは `config/config_finetune.yaml` です.

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune.yaml scripts/finetune.sh
```

### LoRA によるファインチューニング例

LoRAという手法では、もとのモデル（ベースモデル）に追加のパラメーターを付与し、追加のパラメーターのみを訓練します.
ベースモデルのパラメーターは訓練対象としないため、ファインチューニングで必要なメモリ量を抑え、高速に訓練することができます. また、訓練対象のパラメーター数が非常に少ないにも関わらず、通常のファインチューニングとほぼ同等の性能が出ることで知られています.

LoRAを使うには、もとのモデルに追加のパラメーターを付与する処理が必要となりますが、[PEFT](https://github.com/huggingface/peft)ライブラリを使うことで、簡単に実装できます.


LoRAのサンプルコードは `src/finetune_lora.py` です. 設定ファイル（`config/config_finetune_lora.yaml`）には、ファインチューニングの設定に加え、LoRA特有の設定が含まれています.

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune_lora.yaml scripts/finetune_lora.sh
```

### 補足: PEFT モデルのロードとマージについて

`save_pretrained` で保存した PEFT モデルは更新したモデルパラメータのみを保持しています. そのため、訓練したモデルを利用する際には、事前にベースモデルをロードしておくか、ベースモデルに PEFT で訓練したパラメータをマージし、そのモデルを保存しておく必要があります. ベースモデルへのパラメータのマージと保存は以下のようなコードで実現できます.


```python
...

peft_model = get_peft_model(model)
... # do train

# 訓練結果がベースモデル（model）に反映される
peft_model.merge_and_unload()
model.save_pretrained(<save path>)

```

ベースモデルをロードした後で、訓練済みの PEFT の重みを追加する場合は、以下のようなコードになります

```python
...
# PEFT ライブラリを利用する
import peft


BASE_MODEL_NAME = "cerebras/Cerebras-GPT-256M"
MODEL_NAME = "/scratch/<user_name>/models/<job_id>/cerebras/Cerebras-GPT-256M"

...
if __name__ == '__main__':
    # トークナイザの読み込み（ベースモデル名を指定して読み込む）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # モデルの読み込み
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = peft.PeftModel.from_pretrained(model, MODEL_NAME)
    # 文章生成
    print(generate_text(model, tokenizer, "日本で一番高い山は"))
```

## 分散処理

複数のGPUを活用するには、分散処理についての知識が必要です.

### モデルパラレルとデータパラレル

モデルが単一のGPUに収まる場合は、GPUの数だけモデルをロードし、各GPUに別々のデータを読み込ませることで、複数のGPUを効率よく活用できます. これを**データパラレル**と呼びます.

一方で、モデルが単一のGPUに収まらない場合は、モデルを分割して複数GPUに配置する必要があります.  これを**モデルパラレル**と呼びます.
（特に工夫のない）モデルパラレルでは、入力に近い層の処理が終わった後で、データを別のGPUに移し、最終層まで処理をするため、GPUの処理待ちが発生します

![model parallel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)
**ref. https://huggingface.co/docs/transformers/v4.15.0/parallelism**

これを回避するには、上図下部の図の通り、バッチをマイクロバッチに分割し、複数GPUが並列して処理できるようにする方法があり、これを**パイプラインパラレル**と呼びます.ただし、パイプラインパラレルを行うには、モデル自身を修正する必要があり、実装コストが高くなってしまいます

HuggingFace Transformers のモデルも、基本的にパイプラインパラレルには対応していません. 多くのモデルは（パイプラインではないという意味でナイーブな）モデルパラレルに対応しており、 モデルのロード時に `device_map` を指定できるようになっています.

サンプルプログラムの例では、以下の通り `device_map` を設定ファイルで指定可能です. 割り当てるGPUを層ごとに明示することもできます.

```yaml
model:
  # model parallel training が可能なモデルについては、device_map: auto とすることで自動的にモデルを分割して複数GPUに読み込んでくれる
  device_map: auto
  trust_remote_code: true
  torch_dtype: torch.float16
```

## 訓練とメモリ消費量

前述の通り、推論時とくらべて訓練時に必要なメモリは非常に大きくなります. それは、

1. モデルパラメータ
2. 勾配情報
3. オプティマイザの状態

を全て保持しておく必要があるためです. そのため、ここまでの方法では大きめのモデルでの推論はできても、訓練対象のパラメータ数をうまく絞るなどの工夫がないと大きめのモデル（7Bや13B）を訓練することはできません.

しかし、 DeepSpeed を利用すると、無駄なメモリ消費を抑えることができます. また、 DeepSpeed は GPU に収まらなかったパラメータを CPU にオフロードする機能も備えているため、大きなモデルであっても訓練できるようになります

![deepspeed](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

ここからは比較的大きなモデル `cyberagent/open-calm-7b` を利用した例を示します.

### DeepSpeed を用いた Multi GPU 訓練の例

以下の例では、DeepSpeedを利用して複数のGPUを効率的に利用しています.


```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cyberagent/open-calm-7b -v CONFIG=config/config_finetune_lora_deepspeed.yaml scripts/finetune_lora_deepspeed.sh 
```

### DeepSpeed を用いた Multi-Node Multi GPU 訓練（分散訓練）の例

ABCI で DeepSpeed を用いて Multi-Node Multi GPU 訓練を行うには、OpenMPI runner を用いる方法とPDSH runner を用いる方法があります. ここでは OpenMPI runner を用いる方法を紹介します

#### OpenMPI runner を用いる方法

1. `deepspeed` のソースコードを修正します.
   `.venv/lib64/python3.11/site-packages/deepspeed/launcher/multinode_runner.py` の l.144 付近の `eth0` を `eno1` に書き換えます.
   
   書き換え後のコードは以下のようになります.
   ```python
           mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-hostfile',
            f'{self.args.hostfile}',
            '--mca',
            'btl',
            '^openib',
            '--mca',
            'btl_tcp_if_include',
            'eno1',
            # 'eth0',
        ] + split(self.args.launcher_args)
   ```

   DeepSpeed ではデフォルトで[PDSH](https://github.com/chaos/pdsh/)を用いて分散学習を行いますが、ABCIでは ssh先のノードで Python を読み込めずにエラーとなりるようです。launcher として OpenMPI を指定しますが、 OpenMPI launcher では、自動的に `--mca btl_tcp_if_include eth0` というオプションが指定されます. しかし ABCI ではイーサネットインターフェイス名が `eno1` なので、このオプションが無視され、エラーが起きてしまいます. これを回避するために上記の箇所を修正することで、 `--mca btl_tcp_if_include eno1` というオプションが指定されるようにします

2. 訓練の実施

`scripts/finetune_lora_deepspeed_multinode.sh ` で指定しているノード数を変更することで、並列実行する計算ノードの台数を変更できます. 4台と8台など、台数を変えた時に訓練時間がどう変化するか、確認してみてしてください.

```bash
$ qsub -g $GROUP -v MODEL=cyberagent/open-calm-7b -v CONFIG=config/config_finetune_lora_deepspeed.yaml scripts/finetune_lora_deepspeed_multinode.sh 
```

## 最後に（つまりそうなポイント）

1. cache ディレクトリ
   モデルやデータセットは、デフォルトで `~/.cache/huggingface/` 以下にキャッシュされます. ホームディレクトリ以下は利用可能な容量が少ないため、複数のモデルを利用しているとすぐに容量がいっぱいになってしまいます. これを回避するには、以下のようにキャッシュディレクトリを指定します（以下の例では scratch 領域に保存していますが、グループ領域でも問題ないと思います）.
   
   ```bash
   export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
   ```

2. v100 では、load_in_8bit は勾配情報をうまく処理できない
   load_in_8bit を指定すると、勾配が 0 となってしまい訓練が進まないことがあるようです. そのため、今回のサンプルでは load_in_8bit を指定していません.
3. モデルのロードを `with training_args.main_process_first()` で囲み、 `deepspeed` で訓練をしようとすると、モデルのロードから進まなくなります. そのため、サンプルではモデルのロードは `with training_args.main_process_first()` で囲んでいません


以上です！
