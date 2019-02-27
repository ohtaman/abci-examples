#!/bin/bash


module load python/3.6 cuda/9.0 cudnn/7.2 nccl/2.3
python3 -m venv venv 
source venv/bin/activate

pip install -r requirements.txt

