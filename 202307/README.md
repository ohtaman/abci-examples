## ハッカソン用のサンプルリポジトリ

### はじめての ABCI

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
