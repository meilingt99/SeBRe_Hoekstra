#!/usr/bin/env python
# coding: utf-8

# # Developing Brain Atlas through Deep Learning
#
# ## A. Iqbal, R. Khan, T. Karayannis
# # .
# # .
# # .
# # adapted to run in .py format from command line

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

# Set training, validation, and testing image and mask paths
training_images_folder = 'example_training_images'
masks_folder = 'example_training_masks'
validation_images_folder = 'example_val_images'
validation_masks_folder= 'example_val_masks'
testing_images_folder='example_test_images'
testing_masks_folder='example_test_masks'

 ## Configurations

class BrainConfig(Config):
    """Configuration for training on the brain dataset.
    Derives from the base Config class and overrides values specific to the brain dataset.
    """
    # def __init__(self,pixels):
    #     self.IMAGE_MIN_DIM = int(pixels) #128
    #     self.IMAGE_MAX_DIM = int(pixels) #128
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
    STEPS_PER_EPOCH = 9 #2000 #steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
                          #steps_per_epoch = TotalTrainingSamples / TrainingBatchSize (default to use entire training data per epoch; can modify if required)
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 9 #100 #validation_steps = TotalvalidationSamples / ValidationBatchSize
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

# ## Dataset
#
# Load training dataset
#
# Extend the Dataset class and add a method to load the brain sections dataset, `load_brain()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference() # do not need to for now

# ### change directory here where the dataset is located
# ### .
# ### .

########### Create training dataset:

class BrainDataset_Train(utils.Dataset):
    """Generates the brain section dataset. The dataset consists of locally stored
    brain section images, to which file access is required.
    """
    #see utils.py for default def load_image() function; modify according to your dataset
    def load_brain(self):
        """
        for naming image files follow this convention: '*_(image_id).png'
        """
        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        self.add_class('brain_cortex','1','cortex')
        os.chdir(training_images_folder)
        im_id = 0
        cwd = os.getcwd()
        img_list = glob.glob('*.png') #replace all instance of .jpg with .png
        img_list = natsorted(img_list, key=lambda y: y.lower())
        for i in img_list:  #image_ids start at 0 (to keep correspondence with load_mask which begins at image_id=0)!
            img = skimage.io.imread(i) #grayscale = 0
            im_dims = np.shape(img)
            self.add_image("brain_cortex", image_id=im_id, path = cwd+'/'+glob.glob('*_'+str(im_id)+'.png')[0],height = im_dims[0], width = im_dims[1])#, depth = im_dims[2])
            im_id += 1

    def load_mask(self,image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""

        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        print("image_id: ",image_id,'*_'+str(image_id))
        os.chdir(masks_folder)
        subfolder = glob.glob('*_'+str(image_id))[0]
        print(subfolder)
        os.chdir(subfolder)

        info = self.image_info[image_id]
        print(info)
        mk_list = glob.glob('*.png')
        print(mk_list)
        count = len(mk_list)
        mk_id = 0
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        print("mask shape: {}".format(np.shape(mask)))
        class_ids = np.zeros(count)

        for m in mk_list:
            bin_mask = skimage.io.imread(m,as_grey=True) # grayscale=0
            print("bin_mask shape: {}".format(np.shape(bin_mask)))
            mk_size = np.shape(bin_mask)
            mask[:, :, mk_id]= bin_mask # mask[:, :, mk_id]= bin_mask

            # Map class names to class IDs.
            class_ids[mk_id] = m[-5] #fifth last position from mask_image name = class_id #need to update(range) if class_ids become two/three-digit numbers
            mk_id += 1
        return mask, class_ids.astype(np.int32)

########### Create validation dataset:

class BrainDataset_Val(utils.Dataset):
    """Generates the brain section dataset. The dataset consists of locally stored
    brain section images, to which file access is required.
    """

    #see utils.py for default def load_image() function; modify according to your dataset

    def load_brain(self):
        """
        for naming image files follow this convention: '*_(image_id+1).png'
        """

        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        self.add_class('brain_cortex','1','cortex')
        os.chdir(validation_images_folder)
        im_id = 0
        cwd = os.getcwd()
        img_list = glob.glob('*.png')
        img_list = natsorted(img_list, key=lambda y: y.lower())
        print("Constructing validation dataset with image list ")
        print(img_list)
        for i in img_list:  #image_ids start at 0 (to keep correspondence with load_mask which begins at image_id=0)!
            img = skimage.io.imread(i) #grayscale = 0
            im_dims = np.shape(img)
            self.add_image("brain_cortex", image_id=im_id, path = cwd+'/'+glob.glob('*_'+str(im_id)+'.png')[0],height = im_dims[0], width = im_dims[1])#, depth = im_dims[2])
            im_id += 1
            #print(im_dims)

    def load_mask(self,image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""
        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        print("image_id", image_id)
        os.chdir(validation_masks_folder)
        subfolder = glob.glob('*_'+str(image_id))[0]#add 1 to image_id, to get to correct corresponding masks folder for a given image
        os.chdir(subfolder)

        info = self.image_info[image_id]
        mk_list = glob.glob('*.png')
        count = len(mk_list)
        mk_id = 0
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8) #shape based on image id
        class_ids = np.zeros(count)

        for m in mk_list:
            print("loading mask: ",m)
            bin_mask = skimage.io.imread(m,as_grey=True) # 2D array of grayscale mask (bitmap)
            mk_size = np.shape(bin_mask) # Dim of above mask bitmap
            mask[:, :, mk_id]= bin_mask # mask[:, :, mk_id]= bin_mask

            # Map class names to class IDs.
            class_ids[mk_id] = m[-5] #fifth last position from mask_image name = class_id #need to update(range) if class_ids become two/three-digit numbers
            mk_id += 1
        return mask, class_ids.astype(np.int32)


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
        self.add_class('brain_cortex','1','cortex')
        os.chdir(testing_images_folder)
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

    def load_mask(self,image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""

        os.chdir(os.path.join(ROOT_DIR,'myDATASET'))
        print('Trying to load mask ',image_id,' from dataset ',self)
        os.chdir(testing_masks_folder)
        subfolder = glob.glob('*_'+str(image_id))[0]#add 1 to image_id, to get to correct corresponding masks folder for a given image
        os.chdir(subfolder)

        info = self.image_info[image_id]
        mk_list = glob.glob('*.png')
        count = len(mk_list)
        mk_id = 0
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8) #shape based on image id
        class_ids = np.zeros(count)

        for m in mk_list:
            bin_mask = skimage.io.imread(m,as_grey=True) # 2D array of grayscale mask (bitmap)
            mk_size = np.shape(bin_mask) # Dim of above mask bitmap
            mask[:, :, mk_id]= bin_mask # mask[:, :, mk_id]= bin_mask

            # Map class names to class IDs.
            class_ids[mk_id] = m[-5] #fifth last position from mask_image name = class_id #need to update(range) if class_ids become two/three-digit numbers
            mk_id += 1
        return mask, class_ids.astype(np.int32)

class InferenceConfig(BrainConfig):
    GPU_COUNT = 1#4
    IMAGES_PER_GPU = 1 #product of GPU_COUNT and IMAGES_PER_GPU must be same as number of images detected on
    DETECTION_MIN_CONFIDENCE = 0.1

def main():
    parser = argparse.ArgumentParser(description='Check validity of SeBRe_training.py input params')
    parser.add_argument("--epochs", help="pixel parameter",action='store',type=int,default=1)
    parser.add_argument("--output_dir",required=True,help="training image directory",action='store',type=str)
    args = parser.parse_args()
    print("This run's parameters: ")
    print("Epochs: ", args.epochs)
    print("Output Directory: ", args.output_dir)

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    print("ROOT_DIR ",ROOT_DIR)

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs", args.output_dir)
    print('Weights will be saved to:',MODEL_DIR)

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    config = BrainConfig()
    config.display()

    # Training dataset
    dataset_train = BrainDataset_Train()
    dataset_train.load_brain()
    dataset_train.prepare() #does nothing for now

    # Validation dataset
    dataset_val = BrainDataset_Val()
    dataset_val.load_brain()
    dataset_val.prepare()#does nothing for now

    # Testing dataset
    dataset_test = BrainDataset_Test()
    dataset_test.load_brain()
    dataset_test.prepare()#does nothing for now
    # ## Create Model

    # Create model in training mode
    # Unnecessary redundant strategy for parallelization on multiple GPUs
    #import tensorflow as tf
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)


    # ## Training
    #
    # Train in two stages:
    # 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
    #
    # 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.epochs,
                layers='heads') #epochs = 1


    # # Fine tune all layers
    # # Passing layers="all" trains all layers. You can also
    # # pass a regular expression to select which layers to
    # # train by name pattern.
    #model.train(dataset_train, dataset_val,
    #            learning_rate=config.LEARNING_RATE / 10,
    #            epochs=5,
    #            layers="all")#layers="heads" ; epochs = 2


    # # Save weights
    # # Typically not needed because callbacks save after every epoch
    # # Uncomment to save manually
    model_path = os.path.join(MODEL_DIR, "SeBRe_FINAL_WEIGHTS.h5")
    model.keras_model.save_weights(model_path)

    # # Training complete
    #
    #
    #
    # ## Detection and evaluation
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join(MODEL_DIR, "SeBRe_FINAL_WEIGHTS.h5")
    #model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    # assert model_path != "", "/Users/ajoseph/Documents/EddyLab/SeBRe_env/SeBRe/mask_rcnn_brain_0003.h5"
    #assert model_path != "/n/eddyfs01/ebenshirim/SeBRe/logs/mask_rcnn_brain_cortex_0005.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    #Evaluation metrics calculated for val and test datasets:
    #intersection/union
    #VOC-Style mAP @ IoU=0.5
    #Total number of pixels (can be converted to voxels by multiplying by Z-stack interval) for predicted and GT masks

    save_folder='val'
    for dataset in [dataset_val,dataset_test]:
        image_ids = dataset.image_ids# np.random.choice(dataset.image_ids, 19)
        print('image_ids:\n',image_ids)
        APs = []
        overlap_list = []
        voxel_list=[]
        true_voxels=[]
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
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)
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
                AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
                print("AP: ", AP)
                print("Precisions: ", precisions)
                print("Recalls: ", recalls)
                print("Overlaps: ", overlaps)
                APs.append(AP)
                overlap_list.append(overlaps[0][0])
                voxel_list.append(np.sum(r['masks']))
                true_voxels.append(np.sum(gt_mask))
            #saving predicted images
            ax = get_ax(1)
            visualize.display_instances(image, r['rois'], r['masks'],r['class_ids'],dataset.class_names, r['scores'], ax=ax, title="Predictions")
            names.append('results_section_img_'+str(int(image_id))+'.png')
            plt.savefig(object_detect+'/results_section_img_'+str(int(image_id))+'.png')
            plt.close()

        print('Dataset:\n',dataset)
        print("mAP: ", np.mean(APs))
        print('APs:',APs)
        print('mean IoU: ',np.mean(overlap_list))
        print('overlaps:',overlap_list)
        print('predicted voxel list:', voxel_list)
        print('ground truth voxel list:',true_voxels)
        print('total voxels:', np.sum(voxel_list))
        print('total GT voxels:', np.sum(true_voxels))
        df = pd.DataFrame()
        df['image_names'] = names
        df['APs'] = APs
        df['overlaps'] = overlap_list
        df['predicted_voxel_list'] = voxel_list
        df['ground_truth_voxel_list'] = true_voxels
        df.to_csv(os.path.join(object_detect,"final_results.csv"),index=False)





if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    main()
