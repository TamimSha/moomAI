import threading
import cv2
import numpy as np

class ImageLoader(threading.Thread):
    def __init__(self, image_names, path):
        #super().__init__(group, target, name, args, kwargs, *, daemon)
        threading.Thread.__init__(self)
        self.__image_names = image_names
        self.__len = len(self.__image_names)
        self.__path = path
        self.__batch = np.empty(self.__len, dtype=object)
        self.done = False

    def run(self):
        for i in range(0, self.__len):
            self.__batch[i] = np.array(cv2.imread(
                self.__path + self.__image_names[i], -1)).astype(np.uint8)
        self.done = True

    def getBatch(self):
        return self.__batch

