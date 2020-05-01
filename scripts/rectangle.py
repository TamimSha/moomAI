import numpy as np
import math

class Rectangle:
    def __init__(self, width, length):
        self.tl = np.array([0, 0])
        self.tr = np.array([width, 0])
        self.br = np.array([width, length])
        self.bl = np.array([0, length])
        self.rotation = 0

    def getArea(self):
        return 0.5 * np.linalg.norm(self.tl - self.tr) * np.linalg.norm(self.tr - self.br)

    def getRotation(self):
        return self.rotation

    def rotate(self, theta):
        # tl
        pCircle = np.array([math.cos(math.pi * ((135. - self.rotation - theta) / 180.)),
         math.sin(math.pi * ((135. - self.rotation - theta) / 180.))])
        
        
