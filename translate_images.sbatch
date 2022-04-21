#!/bin/bash

## change this job name to reflect your job
#SBATCH -J image_warping

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p shared
#SBATCH --mem 16gb

## SeBRe.logs/ is the output and error directory
#SBATCH -e SeBRe.logs/%A.err
#SBATCH -o SeBRe.logs/%A.out

##load any modules here
module load gcc/7.1.0-fasrc01 ImageMagick/7.0.8.28-fasrc02
module load GCCcore/8.2.0 Python/3.7.2
module load Anaconda3/2019.10

python translate_images.py $1 $2 $3
