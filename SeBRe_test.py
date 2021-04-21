import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import glob #for selecting png files in training images folder
from natsort import natsorted, ns #for sorting filenames in a directory
import skimage
from skimage import io

from config import Config
import utils
import model as modellib
import visualize
from model import log

testing_images_folder='example_test_images'

## Configurations

class BrainConfig(Config):
   """Configuration for training on the brain dataset.
   Derives from the base Config class and overrides values specific to the brain dataset.
   """
   # Give the configuration a recognizable name
   NAME = "brain_cortex"
   # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
   # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
   GPU_COUNT = 4
   IMAGES_PER_GPU = 1 #8 ; reduced to avoid running out of memory when image size increased
   # Number of classes (including background)
   NUM_CLASSES = 1 + 1  # background + 9 regions
   # Use small images for faster training. Set the limits of the small side
   # the large side, and that determines the image shape.
   IMAGE_MIN_DIM = 256 #int(pixels) #128
   IMAGE_MAX_DIM = 256 #int(pixels) #128
   # Use smaller anchors because our image and objects are small
   #top=int(np.log2(int(pixels)/4)+1)
   #RPN_ANCHOR_SCALES = tuple(4*2**np.arange(max(1, top-5),top))
   #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
   #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
   RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
   # Reduce training ROIs per image because the images are small and have
   # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
   TRAIN_ROIS_PER_IMAGE = 32
   # Use a small epoch since the data is simple
   STEPS_PER_EPOCH = 29 #2000 #steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
                         #steps_per_epoch = TotalTrainingSamples / TrainingBatchSize (default to use entire training data per epoch; can modify if required)
   # use small validation steps since the epoch is small
   VALIDATION_STEPS = 2 #100 #validation_steps = TotalvalidationSamples / ValidationBatchSize
                        #Ideally, you use all your validation data at once. If you use only part of your validation data, you will get different metrics for each batch,
                        #what may make you think that your model got worse or better when it actually didn't, you just measured different validation sets.
                        #That's why they suggest validation_steps = uniqueValidationData / batchSize.
                        #Theoretically, you test your entire data every epoch, as you theoretically should also train your entire data every epoch.
                        #https://stackoverflow.com/questions/45943675/meaning-of-validation-steps-in-keras-sequential-fit-generator-parameter-list
   ###### Further changes (experimentation):
    # Maximum number of ground truth instances to use in one image
   MAX_GT_INSTANCES = 1 #100 #decreased to avoid duplicate instances of each brain region
   # DEFAULT
#     MAX_GT_INSTANCES = 8 #100 #decreased to avoid duplicate instances of each brain region
   # Max number of final detections
   DETECTION_MAX_INSTANCES = 1 #100 # #decreased to avoid duplicate instances of each brain region
   MASK_SHAPE = [56, 56]
   # DEFAULT
#     DETECTION_MAX_INSTANCES = 8 #100 # #decreased to avoid duplicate instances of each brain region
   # Minimum probability value to accept a detected instance
   # ROIs below this threshold are skipped
   DETECTION_MIN_CONFIDENCE =  0.75 #0.7 #AJ probably want to think about lowering
   # Non-maximum suppression threshold for detection
   DETECTION_NMS_THRESHOLD = 0.3 # if overlap ratio is greater than the overlap threshold (0.3), suppress object (https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python

# ## Notebook Preferences

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

########### Create testing dataset:

class BrainDataset_Test(utils.Dataset):
    """Generates the brain section dataset. The dataset consists of locally stored
    brain section images, to which file access is required.
    """

    #see utils.py for default def load_image() function; modify according to your dataset

    def load_brain(self):
        """
        for naming image files follow this convention: '*_(image_id+1).png'
        """

        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        os.chdir(testing_images_folder)
        self.add_class('brain_cortex','1','cortex')
        im_id = 0
        cwd = os.getcwd()
        img_list = glob.glob('*.png')
        img_list = natsorted(img_list, key=lambda y: y.lower())
        print("Constructing testing dataset with image list ")
        print(img_list)
        for i in img_list:  #image_ids start at 0 (to keep correspondence with load_mask which begins at image_id=0)!
            img = skimage.io.imread(i) #grayscale = 0
            im_dims = np.shape(img)
            self.add_image("brain_cortex", image_id=im_id, path = cwd+'/'+glob.glob('*_'+str(im_id)+'.png')[0],height = im_dims[0], width = im_dims[1])#, depth = im_dims[2])
            im_id += 1

class InferenceConfig(BrainConfig):
    GPU_COUNT = 1#4
    IMAGES_PER_GPU = 1 #product of GPU_COUNT and IMAGES_PER_GPU must be same as number of images detected on
    DETECTION_MIN_CONFIDENCE = 0.1

def main():
    parser = argparse.ArgumentParser(description='SeBRe_Test.py params: model_dir, output_dir')
    parser.add_argument("--model_dir",required=True,help="model directory",action='store',type=str)
    parser.add_argument("--output_dir",required=True,help="output directory",action='store',type=str)
    args = parser.parse_args()
    ROOT_DIR = os.getcwd()

    # Testing dataset
    dataset_test = BrainDataset_Test()
    dataset_test.load_brain()
    dataset_test.prepare()#does nothing for now
    # ## Create Model

    MODEL_DIR = os.path.join(ROOT_DIR, "logs", args.model_dir)
    model_path = os.path.join(MODEL_DIR, "SeBRe_FINAL_WEIGHTS.h5")
    print("Model Dir: ", MODEL_DIR)
    print("Model path: ", model_path)

    # ## Detection and evaluation
    inference_config = InferenceConfig()
    inference_config.display()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    save_folder='test'
    for dataset in [dataset_test]:
        image_ids = dataset.image_ids# np.random.choice(dataset.image_ids, 19)
        print('image_ids:\n',image_ids)
        #APs = []
        #overlap_list = []
        voxel_list=[]
        #true_voxels=[]
        object_detect = os.path.join(ROOT_DIR,'logs/',args.output_dir,save_folder)#type[1])
        os.makedirs(object_detect)
        names = []
        save_folder='test'
        print('saving files to: ',object_detect)
        for image_id in image_ids:#changed from range(0,199) to span all image_ids
        # Load image and ground truth data
        #This won't work once there's no GT to go with the testing images
        #modellib.load_image_gt  will have to be changed to image= dataset.load_image(image_id)
        #and the GT evaluation moved within an if loop to apply only to validation set
            #image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)
            image = dataset_test.load_image(image_id)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
                # Run object detection
            #print('image:\n',image)
            #print('\nimagedim:\n',image.shape)
            results = model.detect([image], verbose=5)
            r = results[0]
            #print('results:\n rois\n',r["rois"],'\n class_ids:\n',r["class_ids"],'\nscores:\n' r["scores"],'\nmasks:\n', r['masks'])
            #print('gt_mask:\n',gt_mask,'pred_mask:\n',r['masks'])

            # Compute QC metrics

                #If using with testing data where GT is unknown, this will have to be removed
                #added an if condition to avoid errors due to no mask being predicted
            if len(r['masks'])>0:
                # AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
                # print("AP: ", AP)
                # print("Precisions: ", precisions)
                # print("Recalls: ", recalls)
                # print("Overlaps: ", overlaps)
                # APs.append(AP)
                # overlap_list.append(overlaps[0][0])
                print(r['masks'].shape)
                voxel_list.append(np.sum(r['masks']))
                names.append('results_section_img_'+str(int(image_id))+'.png')
                # print(np.unique(r['masks']))
                # print(r['masks'])
                # print("r Masks shape: ", r['masks'].shape)
                # true_voxels.append(np.sum(gt_mask))
                # print(np.unique(gt_mask))
                # print(gt_mask)
                # print("gt mask shape: ", gt_mask.shape)
            #saving predicted images
            ax = get_ax(1)
            visualize.display_instances(image, r['rois'], r['masks'],r['class_ids'],dataset.class_names, r['scores'], ax=ax, title="Predictions")
            plt.savefig(object_detect+'/results_section_img_'+str(int(image_id))+'.png')
            plt.close()

        print('predicted voxel list:', voxel_list)
        df = pd.DataFrame()
        df['image_names'] = names
        df['predicted_voxel_list'] = voxel_list
        df.to_csv(os.path.join(object_detect,"final_results.csv"),index=False)





if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    main()
