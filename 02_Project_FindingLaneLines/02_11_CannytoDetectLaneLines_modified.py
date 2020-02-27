# doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2  # bring in OpenCV libraries

# Read in the image and convert to grayscale
image = cv2.imread('test_real.png')
plt.imshow(image)
plt.show()
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # grayscale conversion
plt.imshow(gray)
plt.show()
plt.imshow(gray, cmap='gray')
plt.show()

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv.2Canny() applies a 5*5 Gaussian internally
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
image_15 = cv2.GaussianBlur(image, (15, 15), 0)
plt.imshow(blur_gray, cmap='gray')
plt.show()
plt.imshow(image_15)
plt.show()

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
plt.show()
