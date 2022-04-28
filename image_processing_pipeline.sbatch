#!/bin/bash

## change this job name to reflect your job
#SBATCH -J translations

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p shared
#SBATCH --mem 16gb

## SeBRe.logs/ is a directory I made before running this sbatch file
#SBATCH -e SeBRe.logs/%A.err
#SBATCH -o SeBRe.logs/%A.out

##load any modules here
module load gcc/7.1.0-fasrc01 ImageMagick/7.0.8.28-fasrc02
module load GCCcore/8.2.0 Python/3.7.2
module load Anaconda3/2019.10

HOME_DIR=$PWD

##Part 1: Move dataset from preprocessed_datasets to myDATASET
## set DEST_DIR
DEST_DIR=$PWD/myDATASET
## set SOURCE_DIR
SOURCE_DIR=$PWD/preprocessed_datasets

# move all raw images and masks into $DEST_DIR
cp -r $SOURCE_DIR/$1 $DEST_DIR
cd $DEST_DIR/$1

#delete all hidden images
find \( -iname "._*" \) -type f -exec rm "{}" \;

# resize and reformat images
# redefine dimensions (4279x5689) as necessary
find -name "*.png" -exec magick convert {} -define png:color-type=2 -resize '4279x5689' -alpha Opaque {} \; -print
echo "COMPLETED RESIZING"

## Make filler folder:
MASK_INT='cortex_masked'

mkdir $DEST_DIR/$MASK_INT; i=0; for fi in $DEST_DIR/$1/*_mask.png; do mkdir $DEST_DIR/$MASK_INT/masked_section_$i; mv "$fi" $DEST_DIR/$MASK_INT/masked_section_$i/masked_section_$i.png; i=$((i+1)); done
echo "COMPLETED MASKS"

mkdir $DEST_DIR/$2; i=0; for fi in $DEST_DIR/$1/*.png; do mv "$fi" $DEST_DIR/$1/section_img_$i.png; i=$((i+1)); done
echo "COMPLETED IMAGES"

cd $HOME_DIR

source SeBRe_env/bin/activate

python custom_dataset_create.py $MASK_INT $3

rm -r $DEST_DIR/$1
rm -r $DEST_DIR/$MASK_INT
