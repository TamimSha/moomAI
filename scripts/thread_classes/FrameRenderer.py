import cv2
import threading

class FrameRenderer(threading.Thread):
    def __init__(self, name, output, videoFile, start, end):
        threading.Thread.__init__(self)

        self.video = cv2.VideoCapture()
        self.open = False
        self.output = output
        self.__progress = 0.0
        self.name = name
        self.__alive = True
        self.totalFrames = 0
        self.__start = start
        self.__end = end
        self.index = 0

        self.__setVideo(videoFile)

    def __setVideo(self, videoFile):
        try:
            self.video = cv2.VideoCapture(videoFile)
            if self.video.isOpened():
                self.open = True
                self.totalFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            else:
                raise NameError
        except:
            self.kill()
        return self.open

    def getProgress(self):
        return self.index / (self.totalFrames - self.__start - self.__end)

    def kill(self):
        self.open = False
        self.__alive = False
        
    def isAlive(self):
        return self.__alive       

    def run(self):  
        
        while(self.open):
            if self.__alive:
                hasFrame, image = self.video.read()
                if hasFrame:
                    self.index += 1
                    if(self.index > self.__start and self.index < self.totalFrames - self.__end):
                        (h, w) = image.shape[:2]
                        test = image[h//4, w//4].all() == 0 and image[h//4, 3*w//4].all() == 0 and image[3*h//4, 3*w//4].all() == 0 and image[3*h//4, w//4].all() == 0
                        if(not test):
                            cv2.imwrite(self.output+self.name+"_{:05d}".format(self.index)+".jpg", image)
                else:
                    self.__alive = False
            else:
                self.open = False
                self.video.release()
                self.__progress = 100.

