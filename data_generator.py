import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

HEIGHT_LBD = 10
WIDTH_LBD = 10
HEIGHT_UBD = 26
WIDTH_UBD = 26
GRAYSCALE_MAX_VALUE = 255
CHARACTER_PIXEL_NUM_TH = 100
BACKGROUD_PIXEL_NUM_TH = 1000

IMAGE_FILE_DIR = "./images/raw_images"

image_filename = "0020.jpg"
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

# If the character is at the border of the image, it will be found by the cv2.findContours funciton.
# So, after generating the new image, set the top and bottom border to be backgroud.
new_image[0, :] = GRAYSCALE_MAX_VALUE
new_image[-1, :] = GRAYSCALE_MAX_VALUE

plt.imshow(new_image)

_, contours, hierarchy = cv2.findContours(new_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

char_rects = [] # rectangle for characters
for c in contours:
	(x, y, w, h) = cv2.boundingRect(c)
	if h > HEIGHT_LBD and w > WIDTH_LBD and h < HEIGHT_UBD and w < WIDTH_UBD:
		char_rects.append((x, y, w, h))

char_rects.sort(key=lambda rect: rect[0])

fig = plt.figure()
for idx, (x,y,w,h) in enumerate(char_rects):
	img_tmp = new_image[y:y+h, x:x+w]

	# Clean the border
	img_tmp[0, :] = GRAYSCALE_MAX_VALUE
	img_tmp[-1, :] = GRAYSCALE_MAX_VALUE
	img_tmp[:, 0] = GRAYSCALE_MAX_VALUE
	img_tmp[:, -1] = GRAYSCALE_MAX_VALUE

	fig.add_subplot(1, len(char_rects), idx+1)
	plt.imshow(img_tmp)


plt.show()