import cv2
import numpy as np
import json


import sys
sys.path.insert(0, './')
from scripts import helper_tools
from data import data

def run():

    files = data.getData()

    image = cv2.imread(files['output'] + "01__00003.jpg", -1)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 0.5)
    rotated = cv2.warpAffine(image, M, (w, h))
    #displayImage(rotated)



run()