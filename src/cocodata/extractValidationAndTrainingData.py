import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image as im
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--testdata", help="Create test data only", action="store_true")
args = parser.parse_args()

dataDir = '/home/helge/data/coco'
srcDir = os.path.join(dataDir, 'images')
train = os.path.join(srcDir, 'train2017')
validate = os.path.join(srcDir, 'val2017')
test = os.path.join(srcDir, 'test2017')

dstDir = '/home/helge/dev/unet/data/coco'
trainDir = os.path.join(dstDir, 'train')
validateDir = os.path.join(dstDir, 'validate')

testDir = os.path.join(dstDir, 'test/input')
resultDir = os.path.join(dstDir, 'result')

dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

categoryNames = ['person', 'tennis racket', 'sports ball']
# categoryNames = ['tennis racket']

numTrainSamples = 500
numValidateSamples = 200
numTestSamples = 100

# Create target file-structure
categoryEntries = {}
trainMainImageDir = os.path.join(trainDir, "image")
trainMainMaskDir = os.path.join(trainDir, "mask")
validateMainImageDir = os.path.join(validateDir, "image")
validateMainMaskDir = os.path.join(validateDir, "mask")

catids = coco.getCatIds(catNms=categoryNames)
imgIds = coco.getImgIds(catIds=catids)
images = coco.loadImgs(imgIds)
# Create test data first
testImages = []
count = 0
for image in images:
    annIds = coco.getAnnIds(imgIds=image['id'], catIds=catids,
                            iscrowd=None)  # Not sure if I will care to use this now
    file = image['file_name']
    shutil.copyfile(os.path.join(train, file), os.path.join(testDir, file))
    testImages.append(file)
    count += 1
    if count >= numTestSamples :
        break

if (args.testdata) :
    exit(0)

for category in categoryNames:
    trainImDir = os.path.join(trainMainImageDir, category)
    trainMaskDir = os.path.join(trainMainMaskDir, category)
    os.makedirs(trainImDir,exist_ok=True)
    os.makedirs(trainMaskDir,exist_ok=True)

    valImDir = os.path.join(validateMainImageDir, category)
    valMaskDir = os.path.join(validateMainMaskDir, category)
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
    for img in images:
        if img["file_name"] in testImages :
            continue # Do not reuse testimages for other purposes
        totalmask = None
        catId = categoryEntry.get('catId')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catId[0]], iscrowd=None)
        anns = coco.loadAnns(annIds)
        file = img['file_name']
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
