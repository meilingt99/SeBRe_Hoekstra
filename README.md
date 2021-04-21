# SEBRe_Hoekstra Guide

Adapted from SeBRe (Iqbal et. al, 2019) and Mask R-CNN (He et. al, 2017) for use on Peromyscus brains on the Harvard FAS Research Computing cluster. 

If you're trying to train on multiple areas at once, you'll need to modify the code directly - reach out to me (Meiling) for this!

## Directions for Use

Usage: 

1. Clone this directory.

2. Navigate to this directory and make empty log, myDATASET, and SeBRe.logs directories from the command line using `mkdir` (e.g. `mkdir logs`).

3. Copy over the SeBRe_env virtual environment which is needed to run SeBRe since the FASRC cluster does not have all the necessary dependencies. If you have read permissions on eddyfs01, you can copy it over using `cp -r /n/eddyfs01/mthompson/SeBRe_Hoekstra/SeBRe_env <destination>/SeBRe_Hoekstra`.

4. Copy your training, testing, and image and mask folders into SeBRe_Hoekstra/myDATASET (your images and masks should be in different folders). Your mask folder should have folders inside, one folder for each corresponding image called `masked_section_<image number>`, with the mask for the class (i.e. part of the brain) you're training the model to identify in that folder (e.g. for image 17, there will be a folder called masked_section_17 containing a file masked_section_17.png). If you need help getting images in this folder, I have code for this I'll be updating the directory with soon! 

5. Generate binary masks from mask images: `sbatch custom_dataset_create.py --in_folder <mask image input directory name in myDATASET> --out_folder <mask output directory in myDATASET>`

   **Note**: output directory should not already exist.
   Example usage: `sbatch custom_dataset_create.sbatch --in_folder example_custom_dataset_create_input --out_folder example_custom_dataset_create_output `

6. Train a model using your training dataset (with testing at the end and validation at the end of each epoch). **Before** submitting this job, 

   1. Modify the necessary hyperparameters (see "Guide to Hyperparameters") 
   2. Change training, validation, and testing image and mask paths in the SeBRe_training.py file **directly**. These paths are at the top of the SeBRe_training.py file, right under the imports (line 37).
   3. Usage:`sbatch SeBRe_training.sbatch --epochs <epoch number> --output_dir <output directory name>`
      Example: `sbatch SeBRe_training.sbatch --epochs 10 --output_dir example_SeBRe_run_10_epochs`

7. To conduct inference on novel images using a previously trained model ("test"), modify the testing image path in the SeBRe_testing.sbatch file and then enter this on the command line: `sbatch SeBRe_testing.sbatch --model_dir <model directory> --output_dir <output directory name>`

   1. Model directory should be the SeBRe_training.py output directory in SeBRe_Hoekstra/logs/ that contains the trained model e.g. from the example above `example_SeBRe_run_10_epochs`

   2. Output directory is where your testing images and metrics (voxels only) will be stored.

      **Important Note:** SeBRe_test will create masks in the shape of your original images (e.g. for the example images, 4279 x 5689) while the inference at the end of SeBRe_training will create masks in the shape of the compressed image dimensions specified in the IMAGE_MIN_DIM and IMAGE_MAX_DIM parameters (e.g. 256 x 256).

## Important Files and Directories

Directories:

- myDATASET: Where all training, validation, and testing datasets are stored. 
  - There should be a separate folder for images and masks for training, validation, and testing: e.g. example_training_images and example_training_masks. 
  - Masks corresponding with each image need to be stored in their own folders with the naming format of each folder as section_masks_<corresponding image number> (e.g. example_training_masks/section_masks_23). 
- logs: Where the output from each SeBRe_training and SeBRe_testing run is stored; before submitting a job, the user will specify the name of the output directory in the form of a command line argument 
  - For SeBRe_Training runs, each output directory will contain:
    - brain_cortex<date>: a directory with the model's weights in each epoch  
    - epoch_log.csv: a file with the training and validation metrics from each epoch which you can use to tune hyperparameters such as number of epochs, learning rate, etc.
    - SeBRe_FINAL_WEIGHTS.h5: the model's weights with the best performance - these are the weights used in inference after training is complete to get the metrics and results seen in the test and val folders
    - test and val: folders containing test and validation metrics and results in final_results.csv (metrics) and each test and val image with an overlaid mask
- SeBRe.logs: where error and output slurm logs from each job are stored 

Files:

- SeBRe_training.py: python file (originally a Jupyter notebook in Iqbal et. al's SeBRe) used to train the model for a number of epochs and then conduct inference on the testing and validation datasets to evaluate model performance.
  - Command line arguments: 
    - Number of epochs (integer): I recommend starting with around 25 (which should take around 20-30 minutes) and increasing if needed.
    - Output Directory (string): the name of the output directory in logs where metrics and final testing images with overlaid masks are stored.
  - Divided into 3 main sections (file itself is well commented): 
    1. Config class instance (class BrainConfig) where you can directly modify hyperparameters such as image dimensions, steps_per_epoch, etc.
    2. Dataset class instances for the training, validation, and testing datasets (images and masks). This is where you (currently) modify the paths for training, validation, and testing image and mask paths.
    3. Training and testing of the model (no modifications needed). 
- SeBRe_training.sbatch: bash script used to submit the slurm job (no modification needed)
- SeBRe_testing.py: uses trained models generated by SeBRe_training.py to conduct inference on new images.
- SeBRe_testing.sbatch: bash script used to submit the slurm job (no modification needed)
- custom_dataset_create.py: converts mask images into SeBRe binary masks
- custom_dataset_create.sbatch: bash script used to submit the slurm job (no modification needed)

## Guide to Hyperparameters

Main hyperparameters you should modify before submitting your first job on a new dataset:

- STEPS_PER_EPOCH: set to the ceiling of the number of training images divided by the batch size (4) e.g. the example_training_images dataset has 34 images, so we would set steps_per_epoch = ceiling(34/4) = 9.
- VALIDATION_STEPS: set to the number of validation images.
- Epochs: number of passes through the training dataset - this will vary depending on the size of your training dataset. You can use the metrics in epoch_log.csv to try to adjust these, but this metric is usually inaccurate with fewer validation images. 

Some secondary hyperparameters worth noting:

- Image Dimensions (IMAGE_MIN_DIM and IMAGE_MAX_DIM): what size (in pixels) images are compressed to.
- Learning Rate: essentially how quickly the model "learns" (updates weights), no need to modify unless model performance is extremely poor (e.g. slow convergence or extreme divergence).
- Mask Shape: dimensions of masks produced by the model, requires directly modifying model architecture - reach out to Meiling to modify this. 

## Guide To Additional Files

Files:

- SeBRe_env, pycache: contains the python packages used by SeBRe that are not on the RC server. SeBRe_env is the virtual env that is activated whenever we use SeBRe. 
- config.py, model.py, parallel_model.py, utils.py, visualize.py: contains classes used in the SeBRe_training and SeBRe_testing files.
- mask_rcnn_coco.h5: default weights that the model is initialized with. 

Feel free to reach out to me (Meiling) with any questions!

## Upcoming Updates 

- Mask Processing Code
- SeBRe for multiple classes of masks