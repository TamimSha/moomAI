from PIL import Image, ImagePalette
import os

def displayImage(imgArr):
    img = Image.fromarray(imgArr, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.show()

def getImageNames(path):
    files = os.listdir(path)
    imageNames = []
    for file in files:
        if file[-4:] == ".jpg":
            imageNames.append(file)
    return imageNames