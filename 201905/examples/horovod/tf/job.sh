#!/bin/bash

#$-l rt_F=2
#$-j y
#$-cwd


N_GPU=4

# Environment Modules の初期化
source /etc/profile.d/modules.sh
# Environment Modules の設定
module load openmpi python/3.6 cuda/9.0 cudnn/7.2 nccl/2.3
python3 -m venv venv 
source venv/bin/activate

# プログラムの実行
mpirun -N $N_GPU python3 mnist.py

