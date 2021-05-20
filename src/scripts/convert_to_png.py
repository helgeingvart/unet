import matplotlib.pyplot as plt
import os
from pydicom import dcmread

import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")

args = vars(ap.parse_args())
dir = args.get('dataset')

plt.gray()
dirList = os.listdir(dir)
dirList.sort()

destinationDir = os.path.join(dir, "png")
if not os.path.exists(destinationDir):
    os.makedirs(destinationDir)

for filename in dirList:
    if filename.endswith(".dcm"):

        path = os.path.join(dir, filename)
        destPath = os.path.join(destinationDir, filename)
        ds = dcmread(path)
        array = ds.pixel_array

        base = os.path.splitext(destPath)[0]
        plt.imsave(base + '.png', array)
        print('Processing: ' + path)
