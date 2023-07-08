#!/bin/bash
#$ -l rt_F=4
#$ -l h_rt=6:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N finetune_lora_deepspeed_multinode
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

hostfile=$(mktemp)
for l in `cat $SGE_JOB_HOSTLIST`; do echo $l slots=4; done > $hostfile
trap "rm $hostfile" EXIT
trap "trap - EXIT; rm $hostifle; exit -1" INT PIPE TERM

MASTER_ADDR=$HOSNAME deepspeed \
  --master_addr $HOSTNAME \
  --hostfile $hostfile \
  --no_ssh_check \
  --launcher OpenMPI \
  src/finetune_lora_distribute.py \
  --model_name $MODEL \
  --config_file $CONFIG
