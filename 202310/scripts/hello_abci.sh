#!/bin/bash
#$ -l rt_G.small=1
#$ -j y
#$ -N hello_abci
#$ -o logs/
#$ -cwd

echo "This is ${USER}'s first job on ABCI!"
