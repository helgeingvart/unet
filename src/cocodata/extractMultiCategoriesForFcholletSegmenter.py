import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as im
import shutil
import argparse

dataDir = '/home/helge/data/coco'
srcDir = os.path.join(dataDir, 'images')

dstDir = '/home/helge/dev/unet/data/coco/fchollet-test'
imageDir = os.path.join(dstDir, 'image')
labelDir = os.path.join(dstDir, 'label')

dataType = 'train2017'
srcDir = os.path.join(srcDir, 'train2017')
# validate = os.path.join(srcDir, 'val2017')
# test = os.path.join(srcDir, 'test2017')

annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

categoryNames = ['person', 'tennis racket', 'sports ball']

numSamples = 3000

catids = coco.getCatIds(catNms=categoryNames)
imgIds = coco.getImgIds(catIds=catids)
images = coco.loadImgs(imgIds)

count = 0

for image in images:
    annIds = coco.getAnnIds(imgIds=image['id'], catIds=catids,
                            iscrowd=None)  # Not sure if I will care to use this now
    file = image['file_name']
    shutil.copyfile(os.path.join(srcDir, file), os.path.join(imageDir, file))
    testImages.append(file)

