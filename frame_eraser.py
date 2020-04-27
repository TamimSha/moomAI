import os
import sys
import time
import cv2
import json
import math
import progressbar
import numpy as np
from PIL import Image, ImagePalette

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

    for batch in range(68, math.floor(len(imageNames) / BATCH_SIZE) + 1):
        print(f"\nBatch: {batch} of {math.floor(len(imageNames) / BATCH_SIZE)}")
        batch_host = np.empty(0, dtype=object)
        threads = []
        start = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        if(end > len(imageNames)):
            end = len(imageNames)
        imageName_Batch = imageNames[start:end]
        batch_length = len(imageName_Batch)
        threadBatch = math.floor(batch_length / NUM_THREADS)
        
        for i in range(0, NUM_THREADS):
            if i == NUM_THREADS - 1:
                imageLoader = ImageLoader(imageName_Batch[i * threadBatch:batch_length], files['output'])
            else:
                imageLoader = ImageLoader(imageName_Batch[i * threadBatch:(i+1) * threadBatch], files['output'])
            threads.append(imageLoader)
        for t in threads:
            t.start()
        done = False
        print("Copying from Disk to RAM")
        with progressbar.ProgressBar(max_value=batch_length) as bar:
            while(not done):
                done = all(t.done == True for t in threads)
                progress = 0
                for t in threads:
                    progress += t.progress
                bar.update(progress)
                time.sleep(0.1)
        
        for t in threads:
            temp = np.copy(t.getBatch())
            batch_host = np.append(batch_host, temp)
        for t in threads:
            t.join()
        print(len(batch_host))
        batch_device = np.zeros_like(batch_host)
        print("Copying from RAM to GPU")
        with progressbar.ProgressBar(max_value=batch_length) as bar:
            for i in range(0, batch_length):
                batch_device[i] = driver.mem_alloc(batch_host[i].nbytes) # pylint: disable=no-member, unsupported-assignment-operation
                driver.memcpy_htod(batch_device[i], batch_host[i]) # pylint: disable=no-member
                bar.update(i)
        
        # CUDA Absolute Image Subtraction
        diffBlock = (8,8,3)
        diffGrid = (90,68,1)

        h_diffImage_int = np.zeros_like(batch_host[0], dtype=np.uint8)
        d_diffImage_int = driver.mem_alloc(h_diffImage_int.nbytes) # pylint: disable=no-member
        getImgDiff = __module.get_function("cuda_GetImgDiff")

        # CUDA Sum Image
        num_block = 2295
        block = (512,1,1)
        grid = (num_block,1,1)

        h_sum = np.zeros(num_block, dtype=np.float)
        d_sum = driver.mem_alloc(h_sum.nbytes) # pylint: disable=no-member
        sumPixels = __module.get_function("cuda_SumPixels")

        # CUDA Int to Float image converstion
        h_diffImage_float = h_diffImage_int.astype(np.float32) # pylint: disable=no-member
        d_diffImage_float = driver.mem_alloc(h_diffImage_float.nbytes) # pylint: disable=no-member
        byteToFloat = __module.get_function("cuda_ByteToFloat")

        
        imagesToDelete = []
        print("Processing")
        pixelSum = 0
        with progressbar.ProgressBar(max_value=batch_length) as bar:
            pivot = 0
            threshold = 1.0e+35
            for i in range(0, batch_length - 1):
                getImgDiff(d_diffImage_int, batch_device[pivot], batch_device[i+1], block=diffBlock, grid=diffGrid)
                byteToFloat(d_diffImage_float, d_diffImage_int, block=block, grid=grid)
                sumPixels(d_diffImage_float, d_sum, block=block, grid=grid)
                driver.memcpy_dtoh(h_sum, d_sum) # pylint: disable=no-member
                pixelSum = h_sum.sum()

                if(pixelSum > threshold):
                    pivot = i
                else:
                    imagesToDelete.append(i)
                bar.update(i)

        for i in imagesToDelete:
            #os.remove(files['output']+imageName_Batch[i])
            pass
        print(imageName_Batch[-1])
        print(f'Deleted: {len(imagesToDelete)} images')
        
        #getImgDiff(d_diffImage_int, batch_device[1000], batch_device[1001], block=diffBlock, grid=diffGrid)
        #driver.memcpy_dtoh(h_diffImage_int, d_diffImage_int)
        #displayImage(h_diffImage_int)
        #byteToFloat(d_diffImage_float, d_diffImage_int, block=block, grid=grid)

        #if batch >= 5:
        #    return
        
    return 0

def __getImageNames(path):
    files = os.listdir(path)
    imageNames = []
    for file in files:
        if file[-4:] == ".jpg":
            imageNames.append(file)
    return imageNames

def displayImage(imgArr):
    img = Image.fromarray(imgArr, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.show()

frame_eraser()