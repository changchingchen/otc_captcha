import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


GRAYSCALE_MAX_VALUE = 255
CHARACTER_PIXEL_NUM_TH = 100
BACKGROUD_PIXEL_NUM_TH = 1000

IMAGE_FILE_DIR = "./images/raw_images"

image_filename = "0000.jpg"
image_file_path = os.path.join(IMAGE_FILE_DIR, image_filename)
image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

# Use the histogram of colors to find out the colors for all characters
# The grayscale color bin that contains more than 1000 pixels is the backgroud color
# If a color contains more than 100 pixels, then it is a color of characters 
color_bins = np.arange(GRAYSCALE_MAX_VALUE + 1)
hist, _ = np.histogram(image, color_bins)

colors = np.nonzero((hist > CHARACTER_PIXEL_NUM_TH) & (hist < BACKGROUD_PIXEL_NUM_TH))
new_image = np.ones(image.shape, image.dtype) * GRAYSCALE_MAX_VALUE

for color in colors[0]:
	new_image[image == color] = 0 # Set character to black

plt.imshow(new_image)
plt.show()