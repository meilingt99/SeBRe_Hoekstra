import os
import subprocess
import sys
import numpy as np

import glob #for selecting png files in training images folder
in_folder = sys.argv[1]
out_folder = sys.argv[2]
pixels = int(sys.argv[3])

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

os.chdir(in_folder)
print(os.getcwd())
img_list = glob.glob('*_mask.png') #replace all instance of .jpg with .png
print("image list: ",img_list)
print(os.getcwd())
for i in img_list:
    i_img = i[:len(i)-9] + ".png" # image name (i is mask name)
    commandstrings = []
    commandstrings.append("convert " + str(i) + " -page +0+0 -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/warp0-" + str(i))
    commandstrings.append("convert " + str(i_img) + " -page +0+0 -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/warp0-" + str(i_img))
    # inward warping
    warping_vals_X = [pixels]
    warping_vals_Y = [pixels]
    for val in warping_vals_X:
        # inward X warping
        commandstrings.append("convert " + str(i) + " -resize " + str(4279 - val) + "x5689! PNG48:" + out_folder + "/warpX-in" + str(val) + "-" + str(i))
        commandstrings.append("convert " + str(i_img) + " -resize " + str(4279 - val) + "x5689! PNG48:" + out_folder + "/warpX-in" + str(val) + "-" + str(i_img))
        commandstrings.append("convert " + out_folder + "/warpX-in" + str(val) + "-" + str(i) + " -page 4279x5689+" + str(int(val/2)) + "+0 -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/warpX-in" + str(val) + "-" + str(i))
        commandstrings.append("convert " + out_folder + "/warpX-in" + str(val) + "-" + str(i_img) + " -page 4279x5689+" + str(int(val/2)) + "+0 -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/warpX-in" + str(val) + "-" + str(i_img))
        # outward X warping
        commandstrings.append("convert " + str(i) + " -resize " + str(4279 + val) + "x5689! PNG48:" + out_folder + "/warpX-out" + str(val) + "-" + str(i))
        commandstrings.append("convert " + str(i_img) + " -resize " + str(4279 + val) + "x5689! PNG48:" + out_folder + "/warpX-out" + str(val) + "-" + str(i_img))
        commandstrings.append("convert " + out_folder + "/warpX-out" + str(val) + "-" + str(i) + " -page 4279x5689+" + str(int(val/2)) + "+0 -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/warpX-out" + str(val) + "-" + str(i))
        commandstrings.append("convert " + out_folder + "/warpX-out" + str(val) + "-" + str(i_img) + " -page 4279x5689+" + str(int(val/2)) + "+0 -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/warpX-out" + str(val) + "-" + str(i_img))
    for val in warping_vals_Y:
        # inward Y warping
        commandstrings.append("convert " + str(i) + " -resize 4279x" + str(5689 - val) + "! PNG48:" + out_folder + "/warpY-in" + str(val) + "-" + str(i))
        commandstrings.append("convert " + str(i_img) + " -resize 4279x" + str(5689 - val) + "! PNG48:" + out_folder + "/warpY-in" + str(val) + "-" + str(i_img))
        commandstrings.append("convert " + out_folder + "/warpY-in" + str(val) + "-" + str(i) + " -page 4279x5689+0+" + str(int(val/2)) + " -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/warpY-in" + str(val) + "-" + str(i))
        commandstrings.append("convert " + out_folder + "/warpY-in" + str(val) + "-" + str(i_img) + " -page 4279x5689+0+" + str(int(val/2)) + " -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/warpY-in" + str(val) + "-" + str(i_img))
        # outward Y warping
        commandstrings.append("convert " + str(i) + " -resize 4279x" + str(5689 + val) + "! PNG48:" + out_folder + "/warpY-out" + str(val) + "-" + str(i))
        commandstrings.append("convert " + str(i_img) + " -resize 4279x" + str(5689 + val) + "! PNG48:" + out_folder + "/warpY-out" + str(val) + "-" + str(i_img))
        commandstrings.append("convert " + out_folder + "/warpY-out" + str(val) + "-" + str(i) + " -page 4279x5689+0+" + str(int(val/2)) + " -background \"rgb(0,0,0)\" -flatten PNG48:" + out_folder + "/warpY-out" + str(val) + "-" + str(i))
        commandstrings.append("convert " + out_folder + "/warpY-out" + str(val) + "-" + str(i_img) + " -page 4279x5689+0+" + str(int(val/2)) + " -background \"rgb(237,237,237)\" -flatten PNG48:" + out_folder + "/warpY-out" + str(val) + "-" + str(i_img))
    print(commandstrings)
    for c in commandstrings:
        os.system(c)
