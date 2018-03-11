import os
import re

from sklearn.externals import joblib
import cv2
import numpy as np


UNLABELED_DATA_PATH = "./images/unlabeled_images"
TEMP_LABELED_DATA_PATH = "./images/temp_labeled_images"
CLASSIFIER_FILE_PATH = "./classifiers"
CLASSIFIRE_NAME = "otc_captcha_svm_clf.pkl"

GRAYSCALE_MAX_VALUE = 255


print "Loading classifier..."
clf = joblib.load(os.path.join(CLASSIFIER_FILE_PATH, CLASSIFIRE_NAME))


print "Loading unlabeled data..."
X_predict = []
unlabeled_files = []
for filename in os.listdir(UNLABELED_DATA_PATH):
	match = re.match("(\d+)_(\d+).jpg", filename)

	if not match:
		continue

	if int(match.group(1)) >= 2000:
		break

	unlabeled_files.append(filename)
	image_file_path = os.path.join(UNLABELED_DATA_PATH, filename)
	image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
	ret, image = cv2.threshold(image, GRAYSCALE_MAX_VALUE//2, GRAYSCALE_MAX_VALUE, cv2.THRESH_BINARY)

	image = image.flatten()
	X_predict.append(image/GRAYSCALE_MAX_VALUE)


print "Predicting unlabeled data..."
X_predict = np.asarray(X_predict)
y_predict = clf.predict(X_predict)
print y_predict


print "Moving data to temp labeled data folders..."
for i in range(len(unlabeled_files)):
	char = y_predict[i]
	filename = unlabeled_files[i]
	orig_file_path = os.path.join(UNLABELED_DATA_PATH, filename)
	new_file_path = os.path.join(TEMP_LABELED_DATA_PATH, char, filename)
	os.rename(orig_file_path, new_file_path)
