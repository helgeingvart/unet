import os
import shutil

testSourceDir = "/home/helge/dev/dicom/src/unet-rib-test/data/Harnverhalt/train/label-black"
testDestDir = "/home/helge/dev/dicom/src/unet-rib-test/data/Harnverhalt/train/label"

dirList = os.listdir(testSourceDir)
dirList.sort()

i = 0
for filename in dirList:
    print("Processing: " + filename)
#    os.rename(os.path.join(labelDir,filename), os.path.join(labelDir, str(i) + ".png"))
#    shutil.copyfile(os.path.join(sourceDir,filename), os.path.join(imageDir,str(i) + ".png"))
    shutil.copyfile(os.path.join(testSourceDir,filename), os.path.join(testDestDir,str(i) + ".png"))
    i+=1
