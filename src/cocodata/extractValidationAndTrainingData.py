import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as im
import shutil

dataDir = '/home/helge/data/coco'
srcDir = os.path.join(dataDir, 'images')
train = os.path.join(srcDir, 'train2017')
validate = os.path.join(srcDir, 'val2017')
test = os.path.join(srcDir, 'test2017')

dstDir = '/home/helge/dev/unet/data/coco'
trainDir = os.path.join(dstDir, 'train')
validateDir = os.path.join(dstDir, 'validate')

# trainMaskDir = os.path.join(trainDir,'mask')
# trainImgDir = os.path.join(trainDir,'image')
# valImgDir = os.path.join(validateDir,'image')
# valMaskDir = os.path.join(validateDir,'mask')
testDir = os.path.join(dstDir, 'test/input')
resultDir = os.path.join(dstDir, 'result')

dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# catIds = coco.getCatIds(catNms=['tennis racket'])
categoryNames = ['person', 'chair', 'tv']
# catIds = coco.getCatIds(catNms=categoryNames)
# imgIds = coco.getImgIds(catIds=catIds)

numTrainSamples = 100
numValidateSamples = 50
numTestSamples = 10

# images = coco.loadImgs(imgIds)

# Create target file-structure
categoryEntries = {}
for category in categoryNames:

    categoryDir = os.path.join(trainDir, category)
    trainImDir = os.path.join(categoryDir, "image")
    trainMaskDir = os.path.join(categoryDir, "mask")
    os.makedirs(trainImDir,exist_ok=True)
    os.makedirs(trainMaskDir,exist_ok=True)

    categoryDir = os.path.join(validateDir, category)
    valImDir = os.path.join(categoryDir, "image")
    valMaskDir = os.path.join(categoryDir, "mask")
    os.makedirs(valImDir,exist_ok=True)
    os.makedirs(valMaskDir,exist_ok=True)

    id = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=[id[0]])
    categoryEntries[category] = {'catId': id, 'imgIds': imgIds, 'trainImages': trainImDir, 'trainMasks': trainMaskDir,
                              'valImages': valImDir, 'valMasks': valMaskDir}

for categoryName in categoryEntries:
    categoryEntry = categoryEntries[categoryName]
    imgIds = categoryEntry.get('imgIds')
    images = coco.loadImgs(imgIds)
    count = 0
    for imgPath in images:
        totalmask = None

        catId = categoryEntry.get('catId')
        # annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catId], iscrowd=None)
        annIds = coco.getAnnIds(imgIds=imgPath['id'], catIds=[catId[0]], iscrowd=None)
        anns = coco.loadAnns(annIds)
        file = imgPath['file_name']
        for ann in anns:
            mask = coco.annToMask(ann)
            mask = np.array(mask)
            mask = mask * 255
            totalmask = mask if (totalmask is None) else totalmask + mask
            maskImg = im.fromarray(totalmask)
        if (count < numTrainSamples):
            shutil.copyfile(os.path.join(train, file), os.path.join(categoryEntry.get('trainImages'), file))
            maskImg.save(os.path.join(categoryEntry.get('trainMasks'), file))
        elif (count < numTrainSamples + numValidateSamples):
            shutil.copyfile(os.path.join(train, file), os.path.join(categoryEntry.get('valImages'), file))
            maskImg.save(os.path.join(categoryEntry.get('valMasks'), file))
        # elif (count < numTrainSamples + numValidateSamples + numTestSamples):
        #     index = count - (numTrainSamples + numValidateSamples);
        #     indexString = str(index)
        #     if (index < 10): indexString = '0' + indexString
        #     if (index < 100): indexString = '0' + indexString
        #     shutil.copyfile(os.path.join(train, file), os.path.join(testDir, indexString + ".jpg"))
        #     maskImg.save(os.path.join(resultDir, indexString + ".jpg"))
        else:
            break
        print(maskImg.size)
        count += 1
