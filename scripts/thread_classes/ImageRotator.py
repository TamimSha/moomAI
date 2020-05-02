import cv2
import threading
import random
from scripts.helper_tools import displayImage

class ImageRotator(threading.Thread):
    def __init__(self, imageNames, path, output, iterations, scales):
        threading.Thread.__init__(self)
        self.__imageNames = imageNames
        self.__path = path
        self.__output = output
        self.__iterations = iterations
        self.__scales = scales
        self.__numImages = len(self.__imageNames)
        self.__progress = 0
        self.__alive = True

    def isAlive(self):
        return self.__alive

    def kill(self):
        self.__alive = False

    def getProgress(self):
        return (self.__progress / self.__numImages) * 100

    def run(self):
        (h, w) = cv2.imread(self.__path + self.__imageNames[0], -1).shape[:2]
        for imageName in self.__imageNames:
            image = cv2.imread(self.__path + imageName, -1)
            i = 0
            while i < self.__iterations:
                s = random.random() + 1
                M = cv2.getRotationMatrix2D((w / 2, h / 2), random.randint(1, 360), 1)
                rotated = cv2.warpAffine(image, M, (w, h))
                
                crop_height = random.randint(0, h - int(h//s))
                crop_width = random.randint(0, w - int(w//s))
                cropped = rotated[crop_height:crop_height+int(h//s), crop_width:crop_width+int(w//s)]
                test = cropped[0, 0].all() == 0 or cropped[int(h//s)-1, 0].all() == 0 or cropped[0, int(w//s)-1].all() == 0 or cropped[int(h//s)-1, int(w//s)-1].all() == 0
                '''
                crop_height = random.randint(0, int(h//2))
                crop_width = random.randint(0, int(w//2))
                cropped = rotated[crop_height:crop_height+int(h//2), crop_width:crop_width+int(w//2)]
                test = cropped[0, 0].all() == 0 or cropped[int(h//2)-1, 0].all() == 0 or cropped[0, int(w//2)-1].all() == 0 or cropped[int(h//2)-1, int(w//2)-1].all() == 0
                '''
                if(not test):
                    cv2.imwrite(self.__output+"test/"+imageName[:-4]+"_"+str(i)+".jpg", cropped)
                    full = cv2.resize(cropped, (w//2, h//2))
                    half = cv2.resize(cropped, (w//4, h//4))
                    quarter = cv2.resize(half, (w//8, h//8))
                    cv2.imwrite(self.__output+"full/"+imageName[:-4]+"_"+str(i)+".jpg", full)
                    cv2.imwrite(self.__output+"half/"+imageName[:-4]+"_"+str(i)+".jpg", half)
                    cv2.imwrite(self.__output+"quarter/"+imageName[:-4]+"_"+str(i)+".jpg", quarter)
                    i += 1
                if(not self.__alive):
                    return
            self.__progress += 1
        self.__progress = 100
        self.__alive = False

