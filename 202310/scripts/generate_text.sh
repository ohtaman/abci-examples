#!/bin/bash
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