#!/bin/bash
#$ -l rt_G.large=1
#$ -l h_rt=6:00:00
#$ -j y
#$ -N finetune_lora_deepspeed
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

deepspeed src/finetune_lora_distribute.py --model_name $MODEL --config_file $CONFIG