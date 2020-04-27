import os
import sys
import time
import cv2
import json
import math
import numpy as np

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

from ImageLoader import ImageLoader

BATCH_SIZE = 0
NUM_THREADS = 0

with open('./src/cuda_kernels.cu', 'r') as file:
    source = file.read()
__module = SourceModule(source)

def frame_eraser():
    try:
        with open('./data/video.json', 'r') as data:
            data = data.read()
        files = json.loads(data)
    except:
        return 0

    global BATCH_SIZE
    BATCH_SIZE = files['batch_size']

    global NUM_THREADS
    NUM_THREADS = 8
    imageNames = __getImageNames(files['output'])
    #batch_host = np.empty((BATCH_SIZE), dtype=object)
    #batch_device = np.empty((BATCH_SIZE), dtype=object)

    for batch in range(0, math.floor(len(imageNames) / BATCH_SIZE)):
        print(batch)
        #batch_host = np.empty(0, dtype=np.uint8)
        batch_host = np.empty(0, dtype=object)
        threads = []
        imageName_Batch = imageNames[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
        threadBatch = math.floor(BATCH_SIZE / NUM_THREADS)
        
        for i in range(0, NUM_THREADS):
            imageLoader = ImageLoader(imageName_Batch[i * threadBatch:(i+1) * threadBatch], files['output'])
            threads.append(imageLoader)
        for t in threads:
            t.start()
        done = False
        while(not done):
            done = all(t.done == True for t in threads)
            time.sleep(0.1)
        
        for t in threads:
            temp = np.copy(t.getBatch())
            batch_host = np.append(batch_host, temp)
        for t in threads:
            t.join()

        batch_device = np.zeros_like(batch_host)
        
        for i in range(0, BATCH_SIZE):
            batch_device[i] = driver.mem_alloc(batch_host[i].nbytes)
            driver.memcpy_htod(batch_device[i], batch_host[i])
        if batch >= 4:
            return
        
    return 0

def __getImageNames(path):
    files = os.listdir(path)
    imageNames = []
    for file in files:
        if file[-4:] == ".jpg":
            imageNames.append(file)
    return imageNames



frame_eraser()