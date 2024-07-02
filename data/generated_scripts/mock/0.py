# mylib.py
from PIL import Image

def generator(count):
    for i in range(count):
        yield i

def image_manipulation(image_path):
    with Image.open(image_path) as img:
        return img.resize((800, 600))