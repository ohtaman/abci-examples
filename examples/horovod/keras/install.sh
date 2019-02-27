#!/bin/bash


module load openmpi/2.1.5 python/3.6/3.6.5 cuda/9.0/9.0.176.4 cudnn/7.2/7.2.1 nccl/2.3/2.3.7-1
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
python3 -m venv venv
source venv/bin/activate

pip install --no-cache-dir -r requirements.txt

