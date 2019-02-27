#!/bin/bash

#$-l rt_F=2
#$-j y
#$-cwd


N_PROCESS=8
HOSTFILE=hosts

# Environment Modules の初期化
source /etc/profile.d/modules.sh
# Environment Modules の設定
module load openmpi python/3.6 cuda/9.0 cudnn/7.2 nccl/2.3
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
python3 -m venv venv
source venv/bin/activate

# MPI用にホストファイルを生成
cut -d " " -f 1 $PE_HOSTFILE | sed 's/$/ cpu=4/' > $HOSTFILE

# プログラムの実行
mpirun  -np $N_PROCESS --hostfile $HOSTFILE  python3 mnist.py

