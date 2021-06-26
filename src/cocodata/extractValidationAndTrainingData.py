import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as im
import shutil

dataDir = '/home/helge/data/coco'
srcDir= os.path.join(dataDir, 'images')
train = os.path.join(srcDir, 'train2017')
validate = os.path.join(srcDir, 'val2017')
test = os.path.join(srcDir, 'test2017')

dstDir = '/home/helge/dev/unet/data/coco'
trainDir = os.path.join(dstDir,'train')
trainMaskDir = os.path.join(trainDir,'mask')
trainImgDir = os.path.join(trainDir,'image')
validateDir = os.path.join(dstDir,'validate')
valImgDir = os.path.join(validateDir,'image')
valMaskDir = os.path.join(validateDir,'mask')
testDir = os.path.join(dstDir,'test/input')
resultDir = os.path.join(dstDir,'result')

dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['tennis racket'])
imgIds = coco.getImgIds(catIds=catIds)

numTrainSamples = 1000
numValidateSamples = 700
numTestSamples = 100

images = coco.loadImgs(imgIds)
count = 0
for img in images:
    totalmask = None
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    file = img['file_name']
    for ann in anns:
        mask = coco.annToMask(ann)
        mask = np.array(mask)
        mask = mask * 255
        totalmask = mask if (totalmask is None) else totalmask + mask
    maskImg = im.fromarray(totalmask)
    if (count < numTrainSamples) :
        shutil.copyfile(os.path.join(train, file), os.path.join(trainImgDir, file))
        maskImg.save(os.path.join(trainMaskDir, file))
    elif (count < numTrainSamples + numValidateSamples) :
        shutil.copyfile(os.path.join(train, file), os.path.join(valImgDir, file))
        maskImg.save(os.path.join(valMaskDir, file))
    elif (count < numTrainSamples + numValidateSamples + numTestSamples) :
        index = count - (numTrainSamples + numValidateSamples); indexString = str(index)
        if (index < 10) : indexString = '0' + indexString
        if (index < 100): indexString = '0' + indexString
        shutil.copyfile(os.path.join(train, file), os.path.join(testDir, indexString + ".jpg"))
        maskImg.save(os.path.join(resultDir, indexString + ".jpg"))
    else :
        break
    print(maskImg.size)
    count+=1
