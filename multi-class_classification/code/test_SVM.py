"""
==================================================================
Multi-Class Image Classification (SVM Testing) using BOW + OpenCV
==================================================================

Description:
------------
This script loads a trained multi-class SVM model and tests it
on a set of labeled images. The images are represented using a
Bag-of-Visual-Words (BOW) descriptor built from SIFT features.

Inputs:
-------
- Pretrained SVM model: 'svm' file
- Vocabulary: 'vocabulary.npy'
- Test images from folders like '145.motorbikes-101_test', etc.

Output:
-------
- Prints predicted class for each test image
- Computes and prints accuracy for each class and overall

Classes:
--------
0 - Motorbikes
1 - School Bus
2 - Touring Bike
3 - Airplanes
4 - Car Side

"""


import cv2 as cv
import numpy as np
import os
from sklearn.metrics import accuracy_score

# ===========================
# STEP 1: Initialization
# ===========================
sift = cv.SIFT_create()
vocabulary = np.load('vocabulary.npy')
svm = cv.ml.SVM_load('svm')

descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

# ===========================
# STEP 2: Load Test Data
# ===========================
test_folders = [
    '145.motorbikes-101_test',
    '178.school-bus_test',
    '224.touring-bike_test',
    '251.airplanes-101_test',
    '252.car-side-101_test'
]

labels_test = []
y_true = []
y_pred = []
paths_tested = []

print("\n==================== SVM Testing Started ====================\n")

for folder in test_folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        if bow_desc is None:
            print(f"[!] Warning: No features found in {path}")
            continue

        _, response = svm.predict(bow_desc)
        prediction = int(response[0][0])

        # Determine ground-truth label from folder name
        if 'motorbikes' in path:
            actual = 0
        elif 'school' in path:
            actual = 1
        elif 'touring' in path:
            actual = 2
        elif 'airplanes' in path:
            actual = 3
        else:
            actual = 4

        y_true.append(actual)
        y_pred.append(prediction)
        paths_tested.append(path)

        class_names = ['Motorbikes', 'School Bus', 'Touring Bike', 'Airplanes', 'Car Side']
        print(f"[✓] {os.path.basename(path):30} → Predicted: {class_names[prediction]:15} | Actual: {class_names[actual]}")

# ===========================
# STEP 3: Accuracy Report
# ===========================
print("\n====================== Classification Report ======================\n")
overall_accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Accuracy per class
for class_id, class_name in enumerate(['Motorbikes', 'School Bus', 'Touring Bike', 'Airplanes', 'Car Side']):
    idx = [i for i, label in enumerate(y_true) if label == class_id]
    if idx:
        acc = accuracy_score(np.array(y_true)[idx], np.array(y_pred)[idx])
        print(f"{class_name:15}: {acc:.4f}")
    else:
        print(f"{class_name:15}: No samples found.")

print("\n======================== Testing Complete =========================\n")

