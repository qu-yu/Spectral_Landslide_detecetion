import warnings
warnings.filterwarnings("ignore")

import cv2, os, collections
import numpy as np
from libtiff import TIFF
import matplotlib.pyplot as plt

for i in os.listdir('label'):
    label_1 = TIFF.open('label\\{}'.format(i), mode='r').read_image()
    label_2 = TIFF.open('result\\{}'.format(i), mode='r').read_image()

    label_1[label_1 == 1] = 255
    label_2[label_2 == 1] = 255

    plt.figure(figsize=(5, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(label_1, label='true')
    plt.subplot(2, 1, 2)
    plt.imshow(label_2, label='pred')
    plt.legend()
    plt.savefig('vis/{}.png'.format(i.split('.')[0]))
