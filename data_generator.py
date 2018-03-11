import os
import re
import cv2
import numpy as np


HEIGHT_LBD = 10
WIDTH_LBD = 10
FINAL_IMAGE_SIZE_LEN = 25
GRAYSCALE_MAX_VALUE = 255
CHARACTER_PIXEL_NUM_TH = 100
BACKGROUD_PIXEL_NUM_TH = 1000

IMAGE_FILE_DIR = "./images/raw_images"
UNLABELED_FILE_DIR = "./images/unlabeled_images"

def resize_with_padding(image, length):
	row = image.shape[0]
	col = image.shape[1]

	# calculate number of zeros to be padded on top/bottom/left/right
	pad_t = (length - row)/2
	pad_b = length - row - pad_t
	pad_l = (length - col)/2
	pad_r = length - col - pad_l

	image = np.pad(image,
		((pad_t, pad_b), (pad_l, pad_r)),
		"constant",
		constant_values=(GRAYSCALE_MAX_VALUE, )
	)

	return image


for image_filename in os.listdir(IMAGE_FILE_DIR):

	if not image_filename.endswith(".jpg"):
		continue

	image_num = re.match("(\d+).jpg", image_filename).group(1)
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

	_, contours, hierarchy = cv2.findContours(new_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	char_rects = [] # rectangle for characters
	for c in contours:
		(x, y, w, h) = cv2.boundingRect(c)
		if h > HEIGHT_LBD and w > WIDTH_LBD and h <= FINAL_IMAGE_SIZE_LEN and w <= FINAL_IMAGE_SIZE_LEN:
			char_rects.append((x, y, w, h))

	char_rects.sort(key=lambda rect: rect[0])

	for idx, (x,y,w,h) in enumerate(char_rects):
		img_tmp = new_image[y:y+h, x:x+w]

		# Clean the border
		img_tmp[0, :] = GRAYSCALE_MAX_VALUE
		img_tmp[-1, :] = GRAYSCALE_MAX_VALUE
		img_tmp[:, 0] = GRAYSCALE_MAX_VALUE
		img_tmp[:, -1] = GRAYSCALE_MAX_VALUE
		img_tmp = resize_with_padding(img_tmp, FINAL_IMAGE_SIZE_LEN)

		cv2.imwrite(os.path.join(UNLABELED_FILE_DIR, "{}_{}.jpg".format(image_num, idx)), img_tmp)
