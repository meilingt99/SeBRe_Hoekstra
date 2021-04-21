#!/bin/bash

## change this job name to reflect your job
#SBATCH -J translations

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p shared
#SBATCH --mem 4gb

## SeBRe.logs/ is a directory I made before running this sbatch file
#SBATCH -e SeBRe.logs/%A.err
#SBATCH -o SeBRe.logs/%A.out

module load GCCcore/8.2.0 Python/3.7.2
module load Anaconda3/2019.10

source SeBRe_env/bin/activate

python custom_dataset_create.py $@
