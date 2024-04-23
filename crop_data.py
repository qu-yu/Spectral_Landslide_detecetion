import warnings
warnings.filterwarnings("ignore")

import cv2, os, collections
import numpy as np
from libtiff import TIFF

idx = 0
for i in os.listdir('image'):

    image = TIFF.open('image\{}'.format(i), mode='r').read_image()
    image_min, image_max = np.min(image), np.max(image)
    image = (image - image_min) / (image_max - image_min)  #归一化
    print(image.shape)

    label = TIFF.open('label\{}'.format(i), mode='r').read_image()
    label_min, label_max = np.min(label), np.max(label)
    label = (label - label_min) / (label_max - label_min)

    print(label.shape)

    a, b = image.shape[:2]
    for j in range(0, a - 256, 256):
        for k in range(0, b - 256, 256):
            img = image[j:j + 256, k:k + 256]
            lab = label[j:j + 256, k:k + 256]

            np.save('seg_img/{}.npy'.format(idx), img)
            np.save('seg_label/{}.npy'.format(idx), lab)
            idx += 1
print(idx)
