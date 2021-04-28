#!/usr/bin/env python
# coding: utf-8

# # Developing Brain Atlas through Deep Learning
#
# ## A. Iqbal, R. Khan, T. Karayannis
# # .
# # .
# # .

# # Creating custom (your own) training and testing dataset for SeBRe

# ## Import libraries

import os
import glob #for selecting png files in training images folder
from natsort import natsorted, ns #for sorting filenames in a directory
import skimage
from skimage import io
import skimage.transform #modified
import numpy as np
import sys
import matplotlib.pyplot as plt # modified, added entire line
import argparse

# Root directory of the project
ROOT_DIR = os.getcwd()

# in_folder = sys.argv[1]
# out_folder = sys.argv[2]

mask_colors = [[255,179,190],[186,147,223],[255,224,0],[68,71,149]]

parser = argparse.ArgumentParser(description='custom_dataset_create.py params: input folder and output folder of masks')
parser.add_argument("--in_folder",required=True,help="mask input directory",action='store',type=str)
parser.add_argument("--out_folder",required=True,help="mask output directory",action='store',type=str)
args = parser.parse_args()


# ## Save training and validation images
# 1. Save training images in the folder 'myDATASET/mrcnn_train_dataset_images', with the naming convention 'section_img_0.png, section_img_1.png, section_img_2.png,...'.
# 2. Save validation images in the folder 'myDATASET/mrcnn_val_dataset_images' , with the naming convention 'section_img_0.png, section_img_1.png, section_img_2.png,...'.
#
# ## Load binary masks in SeBRe-readable-format
# 1. Use an SVG editor (eg. BoxySVG, RectLabel, InkScape etc.) to create masks on top of each brain region of interest (eg. cortex, hippocampus, cerebellum etc.), giving the mask for each distinct region a unique RGB color code (eg. cortex[242,25,60], hippocampus[255,72,151],cerebellum[238,93,255]).
# 2. Reduce the opacity of underlying brain tissue image to zero.
# 3. Export masked brain image into PNG file format; save each masked brain image file into a separate subfolder ('masked_section_0','masked_section_1', etc.), inside the folder 'myDATASET/training_images_masked' or 'myDATASET/validation_images_masked'.

# ### Create regionwise binary masks: TRAINING

print("changing dir")

os.chdir(ROOT_DIR)
s = {}
os.chdir(os.path.join(ROOT_DIR,'myDATASET',args.in_folder)) # FOR CREATING TEST DATASET: os.path.join(ROOT_DIR,'myDATASET\\validation_images_masked'); modified
all_masked_sections = natsorted(glob.glob('*'))
for section_num,section_fold in enumerate(all_masked_sections):
    section_masks_dirname = ROOT_DIR+'/myDATASET/'+args.out_folder+'/section_masks_'+str(section_num) #modified
    if not os.path.exists(section_masks_dirname):
        for i in range(len(mask_colors)):
            print(section_fold)
            os.chdir(section_fold)
            masked_image_orig = skimage.io.imread(glob.glob('*')[0],plugin='pil')
            [s1, s2, s3] = np.shape(masked_image_orig)
            masked_image = skimage.transform.resize(masked_image_orig,(s1,s2,s3))
            print('debug0')
            plt.figure(figsize=[20,20])
    #   	 plt.imshow(masked_image)
            plt.axis('off')
            plt.title('masked_section_0.png')
            [size1, size2, size3] = np.shape(masked_image)
            print('debug1')
            ### Create empty mask for each region:
            cortex_mask = np.zeros([size1,size2])
            print(masked_image.shape)
        ### Extract all regionwise masks (eg. cortex, hippocampus, cerebellum, etc.)
            for index,x in np.ndenumerate(masked_image[:,:,0]):
                if int(masked_image[index[0],index[1],0]*255) != 0:
                    print(int(masked_image[index[0],index[1],0]*255),int(masked_image[index[0],index[1],1]*255),int(masked_image[index[0],index[1],2]*255))
                    print(masked_image[index[0],index[1],0],masked_image[index[0],index[1],1],masked_image[index[0],index[1],2])
                if (abs(mask_colors[i][0] - int(masked_image[index[0],index[1],0]*255)) <= 5) and (abs(mask_colors[i][1] - int(masked_image[index[0],index[1],1]*255)) <= 5) and (abs(mask_colors[i][2] - int(masked_image[index[0],index[1],2]*255)) <= 5):
                    cortex_mask[index[0],index[1]] = 1
        ### Save binary masks to training or validation folder (set image folder paths accordingly below):
            section_masks_dirname = ROOT_DIR+'/myDATASET/'+args.out_folder+'/section_masks_'+str(section_num) #modified
            if not os.path.exists(section_masks_dirname):
                os.makedirs(section_masks_dirname)
            print('debug2')
            skimage.io.imsave(section_masks_dirname+'/section_masks_'+str(section_num)+ '_' + str(i+1) + '.png', cortex_mask)
            plt.close("all")
            os.chdir('..')
