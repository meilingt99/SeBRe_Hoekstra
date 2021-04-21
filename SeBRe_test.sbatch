#!/bin/bash

## change this job name to reflect your job
#SBATCH -J SeBRe_Experiment_1

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 7-00:00
#SBATCH -p gpu
#SBATCH --mem 32gb
#SBATCH --gres=gpu:4
#SBATCH --gpu-freq=high

## SeBRe.logs/ is a directory I made before running this sbatch file
#SBATCH -e SeBRe.logs/%A.err
#SBATCH -o SeBRe.logs/%A.out

##load any modules here
#module load GCCcore/8.2.0 Python/3.7.2
module load Anaconda3/2019.10

source SeBRe_env/bin/activate

#pixels=256
python SeBRe_test.py $@
