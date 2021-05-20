
import matplotlib.pyplot as plt
import imageio
import scipy.ndimage as nd
import os

index = '3'

sourceDir = "/home/helge/dev/dicom/src/unet-rib-test/data/Harnverhalt/test_1602"

image = imageio.imread(os.path.join(sourceDir,index + '.png'))
result = imageio.imread(os.path.join(sourceDir, index + '_predict.png'))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image, 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(image, 'gray')

result = nd.zoom(result,2,order=1)
plt.imshow(result, 'jet', interpolation='none', alpha=0.3)
plt.show()
