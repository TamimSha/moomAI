import cv2
import numpy as np
import random
import progressbar
import time
import os

from .thread_classes.ImageRotator import ImageRotator
from .helper_tools import displayImage, getImageNames
from data.data import getData

def diversifier():

    files = getData()
    if (files == 0):
        return 0
    
    path = files['output']+files['number']
    path_frames = files['path']+files['number']

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+"full/"):
        os.mkdir(path+"full/")
    if not os.path.exists(path+"half/"):
        os.mkdir(path+"half/")
    if not os.path.exists(path+"quarter/"):
        os.mkdir(path+"quarter/")

    imageNames = getImageNames(path_frames)
    threads = []
    NUM_THREADS = 16
    length = len(imageNames)

    for i in range(0, NUM_THREADS):
        imageRotator = ImageRotator(imageNames[i*length//NUM_THREADS:(i+1)*length//NUM_THREADS],
        path_frames, files['output']+files['number'], 3, 2)
        threads.append(imageRotator)
    for t in threads:
        t.start()

    try:
        with progressbar.ProgressBar(max_value=100.0) as bar:
            active = True
            while(active):
                isDead = True
                progress = 0.
                for t in threads:
                    isDead = isDead and not t.isAlive()
                    progress += t.getProgress()
                if(isDead):
                    active = False
                progress = min(round(progress / NUM_THREADS, 2), 100)
                bar.update(progress)
                time.sleep(0.1)
    except KeyboardInterrupt:
        for t in threads:
            t.kill()
            t.join()
        return 0

    for t in threads:
        t.join()

    return 1

'''
image = cv2.imread(files['output'] + "01__00003.jpg", -1)
(h, w) = image.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1)
rotated = cv2.warpAffine(image, M, (w, h))
print(rotated.shape)
displayImage(rotated)
crop_height = random.randint(0, h//2)
crop_width = random.randint(0, w//2)
cropped = rotated[crop_height:crop_height+(h//2), crop_width:crop_width+(w//2)]
print(cropped.shape)
displayImage(cropped)
test = cropped[0, 0].all() == 0 or cropped[h//2-1, 0].all() == 0 or cropped[0, w//2-1].all() == 0 or cropped[h//2-1, w//2-1].all() == 0
if(test):
    print("Has black area")
else:
    scaled = cv2.resize(cropped, (w//4, h//4))
    displayImage(scaled)
    '''