# 第1回大規模言語モデル分散学習ハッカソン

2023/07/06 - 2023/07/14 で開催される[第1回大規模言語モデル分散学習ハッカソン](https://abci.ai/event/2023/06/13/ja_event.html) 用のサンプルプログラムです. 以下を含みます.

1. ABCIの基本的な使い方
2. 仮想環境の構築
3. LLMを用いた文章生成
4. HuggingFace Transformers の Trainer を用いた訓練
   - Full Finetuning
   - Finetuning （最終層）
5. PEFT を用いた訓練
6. DeepSpeed による高速化・省メモリ化

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
This is {acf15478nt}'s first job on ABCI!
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
module load python/3.11 cuda/11.7 cudnn/8.6
```

ロードしたモジュールは `module list` で確認できます.

```bash
$ module list
Currently Loaded Modulefiles:
 1) python/3.11/3.11.2   2) cuda/11.7/11.7.1   3) cudnn/8.6/8.6.0  
```

### Python の仮想環境構築

単一ノードでサンプルプログラムを実行するには、 python および cuda モジュールをロードした上で、以下の容量で仮想環境（venv）を構築します.
この作業はインタラクティブノードで実施可能です. また、一度実施しておけばよく、ジョブごとに実行する必要はありません.

```bash
$ git clone https://github.com/ohtaman/abci-examples.git
$ cd abci-examples/202307
$ module load python/3.11 cuda/11.7 cudnn/8.6
$ python3 -mvenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

### bitsandbytes のインストール

bitsandbytes は言語モデルを 8bit/4bit に量子化し、GPUメモリ圧縮・高速化するライブラリです.
V100 では勾配をうまく取り扱えなくいことがあるようですが、推論では利用可能であり、また DeepSpeed を利用した際に若干高速化されるようです.

pip でもインストール可能ですが、更新が早いので GitHubから直接インストールすることをお勧めします. こちらもインタラクティブノードで実行可能です（仮想環境ん入った状態で実行します）.

```bash
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes

CUDA_VERSION=117 make cuda11x_nomatmul
python setup.py install

```

### データセットの準備

データセットは原則ご自身で用意されたものをご利用いただく想定ですが、先日 Yahoo! JAPAN の後悔した日本語言語理解ベンチマーク [JGLUE](https://techblog.yahoo.co.jp/entry/2022122030379907/) に含まれる質問応答データセット JSQuAD をサンプルファイル用に加工するスクリプトを用意しました.
必要があればこちらをご利用ください. 以下の手順でサンプルプログラム用に整形できます.


```bash
git clone https://github.com/yahoojapan/JGLUE.git

./scripts/process_jsquad.py JGLUE/datasets/jsquad-v1.1/train-v1.1.json data/jsquad-train-v1.1.json 
./scripts/process_jsquad.py JGLUE/datasets/jsquad-v1.1/valid-v1.1.json data/jsquad-valid-v1.1.json 
```


## LLMを用いた文章生成

今回のハッカソンでは、 [HuggingFace](https://huggingface.co/) に登録されているモデルを使って、文章生成やファインチューニングを行います.
事前検証で利用したモデルは以下の通りです（いくつかのモデルは訓練時にサンプルスクリプトを少しいじる必要があるかもしれません. 動作しない場合はお気軽にお声がけください）

|モデル名|URL|備考|
|:--:|:--|:--|
| cerebras-gpt-13b | https://huggingface.co/cerebras/Cerebras-GPT-13B
| Dolly-v2-12b | https://huggingface.co/databricks/dolly-v2-12b|
| Falcon-7B | https://huggingface.co/tiiuae/falcon-7b|
| Japanese-gpt-neox | https://huggingface.co/rinna/japanese-gpt-neox-3.6b|
| MPT-7B | https://huggingface.co/mosaicml/mpt-7b|
| OpenCALM | https://huggingface.co/cyberagent/open-calm-7b|
| OpenLLAMA_12b_600bt | https://huggingface.co/openlm-research/open_llama_13b_600bt|


### 推論の実行

`src/scripts/generate_texts.py` に、与えられたデータセットからランダムに100データを抽出し、後続の文章を生成するスクリプトを用意しました.
たとえば `=databricks/dolly-v2-12b` を利用して文章生成をしたい場合は以下のコマンドを実行します.

```
$ qsub -g $GROUP -v MODEL=databricks/dolly-v2-12b -v CONFIG=config/config_generate_texts.yaml scripts/generate_texts.sh 
```

このサンプルプログラムでは、設定ファイル（上記コマンドのno `config/config_generate_texts.yaml`) の `data.test_file` で指定されているファイルが以下のような json ファイルであることを想定して、"title"、"context"、"question" からプロンプトを生成して、言語モデルに入力しています.

```json
[
   {"title": "hogehoge", "context": "hogehoge", "question": "hogehoge"},
   {"title": "hogehoge", "context": "hogehoge", "question": "hogehoge"},
   ...
]
```

プロンプトの生成ルール（テンプレート）は、設定ファイルの `prompt` のとおりです.

**config/config_generate_texts.yaml**
```yaml
prompt: |-
  タイトル: {title}
  
  文脈: {context}

  設問: {question}

  答え: 
```


詳細はコードをご覧ください.


## 訓練

言語モデルのファインチューニングには、教師なし学習（自己教師あり学習）、教師あり学習、強化学習などいくつかの方法がありますが、ここではシンプルに言語モデルとしての学習（教師なし学習・自己教師あり学習）を対象とします

### 更新対象のパラメータによる分類

事前学習ではなくファインチューニングを行う場合は、どのパラメータを更新するかという自由度が生まれます. 本サンプルでは、以下の3パターンを取り扱います.

1. フルファインチューニング
   - 言語モデルのパラメータを全て訓練対象とします
2. 最終層（もしくは指定した層）のファインチューニング
   - 最終層以外のパラメータを `requires_grad` を `False` とすることで、最終層以外のパラメータを固定して、最終層のみ訓練対象とします
   - 訓練対象のパラメータが減るので、省メモリで訓練可能となります
   - 最終層のパラメータを挿げ替えることで、分類問題など事前訓練とは別のタスクに利用することも可能です
3. LoRA
   - ベースとなるモデルのパラメータに低ランク行列を加算するような構造のモデルを作成し、低ランク行列のみを訓練する手法です
   - 2022年に発表された手法で、非常に少ないパラメータ数で、フルファインチューニングと遜色ない精度を実現しています
   - 論文では、アテンション層のみを対象としていますが、より一般的には、任意の重み行列に対応可能です


### サンプルコードの実行

#### フルファインチューニングの例

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune_fullyaml scripts/finetune.sh
```

#### 最終層のファインチューニングの例

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune.yaml scripts/finetune.sh
```

### LoRA の例

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=cerebras/Cerebras-GPT-256M -v CONFIG=config/config_finetune_lora.yaml scripts/finetune_lora.sh
```

### 補足: PEFT モデルのロードとマージについて

`save_pretrained` で保存した PEFT モデルは更新したモデルパラメータのみを保持しています. そのため、訓練したモデルを利用する際には、事前にベースモデルをロードしておくか、ベースモデルに PEFT で訓練したパラメータをマージし、そのモデルを保存しておく必要があります. ベースモデルへのパラメータのマージと保存は以下のようなコードで実現可能です


```python
...

peft_model = get_peft_model(model)
... # do train

# 訓練結果がベースモデル（model）に反映される
peft_model.merge_and_unload()
model.save_pretrained(<save path>)

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

サンプルプログラムの例では、以下の通り `device_map` を設定ファイルで指定可能です. 割り当てるGPUを明示することもできます.

```yaml
model:
  # model parallel training が可能なモデルについては、device_map: auto とすることで自動的にモデルを分割して複数GPUに読み込んでくれる
  device_map: auto
  # 8bit/4bit training may be incompatible with V100.
  # load_in_8bit: true
  trust_remote_code: true
  torch_dtype: torch.float16
```

## 訓練とメモリ消費量

推論時とくらべ、訓練時に必要なメモリは非常に大きくなります. それは、

1. モデルパラメータ
2. 勾配情報
3. オプティマイザの状態

を全て保持しておく必要があるためです. そのため、ここまでの方法では大きめのモデルでの推論はできても、訓練対象のパラメータ数をうまく絞るなどの工夫がないと大きめのモデル（7Bや13B）を訓練することはできません.

しかし、 DeepSpeed を利用すると、無駄なメモリ消費を抑えることができます. また、 DeepSpeed は GPU に収まらなかったパラメータを CPU にオフロードする機能も備えているため、大きなモデルであっても訓練できるようになります

![deepspeed](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

### DeepSpeed を用いた Multi GPU 学習の例

```bash
$ qsub -g $GROUP -l h_rt=3:00:00 -v MODEL=databricks/dolly-v2-12b -v CONFIG=config/config_finetune_lora_deepspeed.yaml scripts/finetune_lora_deepspeed.sh 
```

### DeepSpeed を用いた Multi-Node Multi GPU 学習（分散学習）の例

**以下の情報は調査中の内容を含みます**

正しく訓練できているか確認できていませんが、以下の方法でマルチノードでの訓練が進むことがわかったので、不確実な情報を含みますが、共有します.

1. `deepspeed` のソースコード修正
   `.venv/lib64/python3.11/site-packages/deepspeed/launcher/multinode_runner.py` の l.144 付近の `eth0` を `eno1` に書き換える. 
   
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

あとは、（最新の abci_examples を pull してもらって）以下のコードを実行します
- [こちらのコミット](https://github.com/ohtaman/abci-examples/commit/50f31d93bf15f2bca0490e1238e810141b3c7983#diff-07488714d19467e6158f106ad5ce2af2cbdc7c4dbbf61c471042b33d43040c48R51)で、 `src/finetune_lora_distribute.py` を修正しているので、`git pull` で最新版にアップデートしてください

```bash
$ qsub -g $GROUP -v MODEL=databricks/dolly-v2-12b -v CONFIG=config/config_finetune_lora_deepspeed.yaml scripts/finetune_lora_deepspeed_multinode.sh 
```

- falcon-40b を訓練させる場合は、 `scripts/finetune_lora_deepspeed_multinode.sh ` で、ノード数を 4 から 8 に変更してください（OOM 回避のため）.

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
