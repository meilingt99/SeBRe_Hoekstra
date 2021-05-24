 #!/bin/bash

## change this job name to reflect your job
#SBATCH -J translations

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p shared
#SBATCH --mem 4gb

## rotations.logs/ is a directory I made before running this sbatch file
#SBATCH -e SeBRe.logs/%A.err
#SBATCH -o SeBRe.logs/%A.out

##change this to index which jobs from a job array file you would like to run

##load any modules here
module load gcc/7.1.0-fasrc01 ImageMagick/7.0.8.28-fasrc02
module load GCCcore/8.2.0 Python/3.7.2
module load Anaconda3/2019.10

SOURCE_DIR=$PWD
## Part 1: Translate Images
python translate_images.py $SOURCE_DIR/myDATASET/$1 $SOURCE_DIR/myDATASET/$2

## Part 2: Move translated dataset
## set ROOT_DIR
ROOT_DIR=$PWD/myDATASET #UPDATE as needed
## set SOURCE_DIR
SOURCE_DIR=$PWD/myDATASET #UPDATE as needed

## Make filler folders:
MASK_INT='cortex_masked'
WHOLE_BRAIN_MASK_INT='whole_brain_masked'

#move all raw images and masks into $ROOT_DIR
cd $ROOT_DIR/$2

#delete all hidden images
find \( -iname "._*" \) -type f -exec rm "{}" \;
find -name "*.png" -exec magick convert {} -define png:color-type=2 -resize '4279x5689' -alpha Opaque {} \; -print
echo "COMPLETED RESIZING"

mkdir $ROOT_DIR/$MASK_INT; i=0; for fi in $ROOT_DIR/$2/*_mask.png; do mkdir $ROOT_DIR/$MASK_INT/masked_section_$i; mv "$fi" $ROOT_DIR/$MASK_INT/masked_section_$i/masked_section_$i.png; i=$((i+1)); done
echo "COMPLETED CORTEX"

mkdir $ROOT_DIR/$WHOLE_BRAIN_MASK_INT; i=0; for fi in $ROOT_DIR/$2/*_whole.png; do mkdir $ROOT_DIR/$WHOLE_BRAIN_MASK_INT/masked_section_$i; mv "$fi" $ROOT_DIR/$WHOLE_BRAIN_MASK_INT/masked_section_$i/masked_section_$i.png; i=$((i+1)); done
echo "COMPLETED WHOLE BRAIN"

mkdir $ROOT_DIR/$3; i=0; for fi in $ROOT_DIR/$2/*.png; do mv "$fi" $ROOT_DIR/$3/section_img_$i.png; i=$((i+1)); done
echo "COMPLETED IMAGES"

cd $SOURCE_DIR

source SeBRe_env/bin/activate

python cropped_custom_dataset_create.py $MASK_INT $4
python cropped_custom_dataset_create.py $WHOLE_BRAIN_MASK_INT $5

rm -r $ROOT_DIR/$2
rm -r $ROOT_DIR/$MASK_INT
rm -r $ROOT_DIR/$WHOLE_BRAIN_MASK_INT
