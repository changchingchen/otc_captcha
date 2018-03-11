import os
import re
from datetime import datetime

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import cv2
import numpy as np


LABELED_DATA_PATH = "./images/labeled_images"
CLASSIFIER_FILE_PATH = "./classifiers"
CLASSIFIRE_NAME = "otc_captcha_svm_clf.pkl"

X = []
y = []
GRAYSCALE_MAX_VALUE = 255

print "Loading labeled data..."
for filename in os.listdir(LABELED_DATA_PATH):

	match = re.match("(\d+)_(\d+)_(\d?|\w?).jpg", filename)
	if not match:
		continue

	image_file_path = os.path.join(LABELED_DATA_PATH, filename)
	image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
	ret, image = cv2.threshold(image, GRAYSCALE_MAX_VALUE//2, GRAYSCALE_MAX_VALUE, cv2.THRESH_BINARY)
	image = image.flatten()
	X.append(image/GRAYSCALE_MAX_VALUE)
	y.append(match.group(3))

X = np.asarray(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print X.shape

start_time = datetime.now()
print "Training data..."
clf = svm.SVC()
clf.fit(X_train, y_train)
print clf
print "Time consumed for training: {}".format(datetime.now() - start_time)

print "Dumping classifier...."
joblib.dump(clf, os.path.join(CLASSIFIER_FILE_PATH, CLASSIFIRE_NAME))

start_time = datetime.now()
print "Predicting test data..."
test_score = clf.score(X_test, y_test)
print test_score
print "Time consumed for testing: {}".format(datetime.now() - start_time)
