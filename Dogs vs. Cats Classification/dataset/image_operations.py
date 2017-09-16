from PIL import Image
from PIL import ImageFilter
from scipy import ndimage
import numpy as np
import os, sys
import glob



class imageOperations(object):
    def __init__(self, path):
        self.__path = path
        self.__total_width, self.__total_height, self.__size = 0, 0, 0

        #reading images
        for file in glob.glob(self.__path):
            try:
                im = Image.open(file)
                img = im.resize((50, 50))
                img = img.convert('L')
                img.save(file)
                im.close()
            except IOError:
                print("can't open more images")


    def create_train_data(self):
        images, labels = [], []
        for file in glob.glob(self.__path):
            try:
                im = Image.open(file)
                file_name = im.filename
                l = file_name.strip("dataset/data/train\\")
                if l[0] == 'c':
                    images.append(np.array(im)), labels.append(1)
                else:
                    images.append(np.array(im)), labels.append(0)
            except IOError:
                print("can't open the image")
        return np.array(images), np.array(labels)

    def create_test_data(self):
        images = []
        for file in glob.glob(self.__path):
            im = Image.open(file)
            images.append(np.array(im))
        return np.array(images)


