#!/bin/bash

#MSUB -l nodes=1:ppn=48
#MSUB -l walltime=00:01:00:00
#MSUB -v MODULE=devel/python/3.6_intel
#MSUB -N V2.2_test2

export KMP_AFFINITY=compact
module load ${MODULE}


export PATH=$PATH:/pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/cuda90/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/cuda90/cuda90/lib64

cd /pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0
source env6/bin/activate



cd /pfs/work6/workspace/scratch/ov0392-KeShi_Prak-0/nnPrakSentiment/scripts

CUDA_VISIBLE_DEVICES=0 python3 compareOptimizationTarget.py
