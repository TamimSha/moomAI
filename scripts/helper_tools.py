from PIL import Image, ImagePalette

def displayImage(imgArr):
    img = Image.fromarray(imgArr, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.show()