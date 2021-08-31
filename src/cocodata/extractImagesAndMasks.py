from anyio.streams import file
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
import shutil
from PIL import Image as im

makeTest = False
loadIndexesFromFile = False
numImages = 300

dataDir = '/home/helge/data/coco'
database = os.path.join(dataDir, 'images/train2017')

imageDir = '/home/helge/dev/unet/data/coco/fchollet-test/image'
maskDir = '/home/helge/dev/unet/data/coco/fchollet-test/label'

testImageDir = '/home/helge/dev/unet/data/coco/test'

# coco database related init
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# initialize COCO api for instance annotations
coco = COCO(annFile)
# cats = coco.loadCats(coco.getCatIds())

imageIds = []
if (loadIndexesFromFile):
    if (makeTest):
        file = open("testimages.txt", "r")
    else:
        file = open("fileindexes.txt", "r")

    for line in file:
        index = int(line)
        imageIds.append(index)
    images = coco.loadImgs(imageIds)
else:
    catIds = coco.getCatIds(catNms=['tennis racket'])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

counter = 0
for image in images:
    file = image['file_name']
    if makeTest:
        shutil.copyfile(os.path.join(database, file), os.path.join(testImageDir, str(counter) + '.jpg'))
    else:
        annIds = coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        totalmask = None
        for ann in anns:
            mask = coco.annToMask(ann)
            mask = np.array(mask)
            # mask = mask * 255
            totalmask = mask if (totalmask is None) else totalmask + mask
        # print(f'Max: {np.max(totalmask)}')
        totalmask[totalmask != 0] = 1  # We are only interested in 0 and 1 as labels.
        totalmask = 1-totalmask # Invert
        # print(f'Max: {np.max(totalmask)}')
        mask = im.fromarray(totalmask)
        shutil.copyfile(os.path.join(database, file), os.path.join(imageDir, file))
        mask.save(os.path.join(maskDir, file))
        print(f'Index {counter}, Image size {mask.size}')

    counter = counter + 1
    if counter >= numImages: break
