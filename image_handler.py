from PIL.ImageFile import ImageFile
import os
import random


def resize_image(image: ImageFile, new_size):
    return image.resize(size=new_size)


def generate_random_cat(file_path="cats/data/train/animal cat british_shorthair"):
    content = os.listdir(file_path)
    return os.path.join(file_path, random.choice(content))



generate_random_cat()

