#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --export==devel/python/3.6_intel
#SBATCH --job-name= V2.2_test

export KMP_AFFINITY=compact
module load ${MODULE}


export PATH=$PATH:/pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/cuda90/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/cuda90/cuda90/lib64

cd /pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0
source env6/bin/activate



cd /pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/nnPrakSentiment/scripts

CUDA_VISIBLE_DEVICES=0 python3 RNTN_v2.py
