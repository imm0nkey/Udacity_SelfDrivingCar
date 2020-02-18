#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 'with dimensions: ', image.shape)
