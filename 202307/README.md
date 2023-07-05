# 第1回大規模言語モデル分散学習ハッカソン

2023/07/06 - 2023/07/14 で開催される[第1回大規模言語モデル分散学習ハッカソン](https://abci.ai/event/2023/06/13/ja_event.html) 用のサンプルプログラムです. 以下を含みます.

1. ABCIの基本的な使い方
   - シンプルな Job スクリプト
2. 仮想環境の構築
3. LLMを用いた文章生成
4. HuggingFace Transformers の Trainer を用いた訓練
   - Full Finetuning
   - Finetuning （最終層）
5. PEFT を用いた訓練
6. DeepSpeed による高速化・省メモリ化

## ABCI の基本的な使い方

ABCI では、

[abci_firststep](./abci_firststep)

### LLMを用いてテキスト生成

[generate_text](./generate_text)

#### 今回利用する言語モデル

HuggingFace に登録されている言語モデルを利用する想定. 以下のモデルで事前検証を行った

|モデル名|URL|備考|
|:--:|:--|:--|
| cerebras-gpt-13b | https://huggingface.co/cerebras/Cerebras-GPT-13B |
| Dolly-v2-12b | https://huggingface.co/databricks/dolly-v2-12b|
| Falcon-7B | https://huggingface.co/tiiuae/falcon-7b|
| Japanese-gpt-neox | https://huggingface.co/rinna/japanese-gpt-neox-3.6b|
| MPT-7B | https://huggingface.co/mosaicml/mpt-7b|
| OpenCALM | https://huggingface.co/cyberagent/open-calm-7b|
| OpenLLAMA_12b_600bt | https://huggingface.co/openlm-research/open_llama_13b_600bt|

#### 今回用いるデータセット

- JGLUE: https://techblog.yahoo.co.jp/entry/2022122030379907/


```bash
git clone https://github.com/yahoojapan/JGLUE.git
cat  \
```

### bitsandbytes のインストール

bitsandbytes を使うと、8bit/4bit に量子化することで、GPUメモリ圧縮や高速化が期待できる
ただし、 V100 では勾配をうまく取り扱えなくなる模様で、簡単には使えない

インストールしていると、 deepspeed での訓練スピードが若干（5%くらい）向上する様子

pip でもインストールできるが、この辺りのライブラリは進化が早いので、githubから直接インストールする.
もちろんxxxしても良い

```bash
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes

CUDA_VERSION=117 make cuda11x_nomatmul
python setup.py install

```


### 訓練

#### 訓練の種類

パラメータの更新方法による分類
1. フルファインチューニング
2. ファインチューニング
   - 最終層のみ更新
   - 入力・出力のみ更新
   - 一部embeddinngを固定して、更新
3. LoRA

訓練方法による分類

1. 言語モデルとしての訓練
2. インストラクションチューンイング
3. RLHF

### ファインチューニング

1. フルファインチューニング
2. 更新する層を限定したファインチューニング
3. LoRAによるファインチューニング
4. DeepSpeedによる並列訓練

### PEFT モデルのロードとマージ

## つまりポイント

1. cache ディレクトリ
   - scratch もしくは group 領域を使う
2. load_in_8bit は勾配情報をうまく処理できない V100
   - とはいえ、最終そうのFNとかだと関係ないはず...何かやりようがあるかも
   - bitsandbytes を使うと若干早くなる
3. モデルのロードを with training_args.main_process_first() で囲むと dead lock が起こっていそう
4. まずは小さいモデルで試してみるとよさそう
