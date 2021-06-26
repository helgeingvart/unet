from anyio.streams import file
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
import shutil
from PIL import Image as im

makeTest = True

dataDir='/home/helge/data/coco'
database = os.path.join(dataDir,'images/train2017')
trainImageDir = '/home/helge/dev/unet/data/coco/train/image'
maskTrainDir = '/home/helge/dev/unet/data/coco/train/label'

testImageDir = '/home/helge/dev/unet/data/coco/test'

# coco database related init
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)
#cats = coco.loadCats(coco.getCatIds())

imageIds = []
if (makeTest) :
    file = open("testimages.txt", "r")
else :
    file = open("fileindexes.txt", "r")

for line in file:
    index = int(line)
    imageIds.append(index)

images = coco.loadImgs(imageIds)
catIds = coco.getCatIds(catNms=['tennis racket']);
counter = 0
for image in images:
    file = image['file_name']

    if (makeTest) :
        shutil.copyfile(os.path.join(database, file), os.path.join(testImageDir, str(counter) + '.jpg'))
        counter = counter + 1
    else :
        annIds = coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        totalmask = None
        for ann in anns:
            mask = coco.annToMask(ann)
            mask = np.array(mask)
            mask = mask * 255
            totalmask = mask if (totalmask is None) else totalmask + mask

        print('Maximum {}', totalmask.max())
        image = im.fromarray(totalmask)
    #    if (image.size==(640,480)) :
        shutil.copyfile(os.path.join(database, file), os.path.join(trainImageDir, file))
        image.save(os.path.join(maskTrainDir, file))
        print(image.size)








    # I = io.imread(image['coco_url'])
    # rows = np.size(I,0);
    # cols = np.size(I,1);
    # print('Image size: ({}, {})'.format(rows,cols))





