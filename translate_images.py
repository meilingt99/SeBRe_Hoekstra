import os
import subprocess
import sys
import numpy as np

import glob #for selecting png files in training images folder
in_folder = os.getcwd() + "/preprocessed_datasets/" + sys.argv[1]
print(in_folder)
out_folder = os.getcwd() + "/preprocessed_datasets/" + sys.argv[2]
print(out_folder)
pixels = int(sys.argv[3])

print(pixels)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


os.chdir(in_folder)
print(os.getcwd())
img_list = glob.glob('*_mask.png') #replace all instance of .jpg with .png
print("image list: ",img_list)
print(os.getcwd())
for i in img_list:
    i_img = i[:len(i)-9] + ".png" # image name (i is mask name)
    # i_whole_brain = i[:len(i)-9] + "_whole.png" # whole brain mask
    commandstrings = []
    commandstrings.append("convert " + str(i) + " -page +0+0 -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/crop0-" + str(i))
    commandstrings.append("convert " + str(i_img) + " -page +0+0 -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/crop0-" + str(i_img))
    # commandstrings.append("convert " + str(i_whole_brain) + " -page +0+0 -background white -flatten PNG48:" + out_folder + "/crop0-" + str(i_whole_brain))
    x_vals = [-1 * pixels,pixels,-1 * pixels,pixels]
    y_vals = [-1 * pixels,-1 * pixels,pixels,pixels]
    for j in range(4):
        x_val = x_vals[j]
        y_val = y_vals[j]
        # x_val = np.random.uniform(-800,800)
        # y_val = np.random.uniform(-800,800)
        commandstrings.append("convert " + str(i) + " -page +" + str(x_val) + "+" + str(y_val) + " -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/crop" + str(j+1) + "-" + str(i))
        commandstrings.append("convert " + str(i_img) + " -page +" + str(x_val) + "+" + str(y_val) + " -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/crop" + str(j+1) + "-" + str(i_img))
        # commandstrings.append("convert " + str(i_whole_brain) + " -page +" + str(x_val) + "+" + str(y_val) + " -background white -flatten PNG48:" + out_folder + "/crop" + str(j+1) + "-" + str(i_whole_brain))
    for c in commandstrings:
        os.system(c)
