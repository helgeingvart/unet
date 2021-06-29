import matplotlib.pyplot as plt
import imageio
import numpy

import scipy.ndimage as nd
from PIL import Image
import os

# sourceDir = "/home/helge/dev/unet/data/coco/result"
sourceDir = "/home/helge/dev/unet/data/coco/result_28.06.21"
testDir = "/home/helge/dev/unet/data/coco/test/input"

# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

plt.ion()

for i in range(99) :
    index = str(i)
    indexRes = index
    if (i < 10) :
        index = "0" + index
    if (i < 100) :
        index = "0" + index
    image = Image.open(os.path.join(testDir,index + '.jpg'))
    result = Image.open(os.path.join(sourceDir, indexRes + '_predict.png'))

    reshaped = result.resize(image.size, Image.BICUBIC)
    result = numpy.array(reshaped)
    image = numpy.array(image)

    # plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, 'gray', interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(image)

# result = nd.zoom(result,2,order=1)
    plt.imshow(result, 'jet', interpolation='none', alpha=0.8)

    plt.show()
    plt.waitforbuttonpress()
    plt.clf()

