# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir='/home/helge/data/coco'
#dataType='val2017'
dataType='train2017'
#annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
#on','dog','skateboard']);
catIds = coco.getCatIds(catNms=['tennis racket']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])

images = coco.loadImgs(imgIds)
for image in images:
    I = io.imread(image['coco_url'])
    rows = np.size(I,0);
    cols = np.size(I,1);
    print('Image size: ({}, {})'.format(rows,cols))

for id in imgIds:
    coco.loadImgs([id])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#img = coco.loadImgs([208597])[0];

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')

# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

mask = coco.annToMask(anns[0]) ## Yeah, that binary mask is finally there :-)


coco.showAnns(anns)
plt.show()

