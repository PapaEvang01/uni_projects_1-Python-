"""
=====================================================================
Image Classification Comparison: k-NN vs SVM using BOW + SIFT + OpenCV
=====================================================================

Description:
------------
This Python script compares the performance of two classifiers (k-NN and SVM)
for a multi-class image classification task using the Bag of Visual Words (BOW)
representation built from SIFT features.

Methodology:
------------
1. Loads precomputed data:
   - BOW descriptors (`index.npy`)
   - Vocabulary (`vocabulary.npy`)
   - Image paths (`paths.npy`)
   - Trained SVM model (`svm`)
2. Assigns labels based on image file paths.
3. Tests both classifiers:
   - Custom implementation of k-NN (with majority voting)
   - OpenCV's SVM (one-vs-all strategy)
4. Evaluates and compares their performance:
   - Prints per-class and overall accuracy.
   - Displays confusion matrices using seaborn.

Inputs:
-------
- index.npy         → BOW descriptors for training images
- vocabulary.npy    → Clustered visual vocabulary (KMeans)
- paths.npy         → Corresponding image paths
- svm               → Pretrained SVM model
- *_test folders    → Test images grouped by class

Outputs:
--------
- Accuracy per class for both classifiers
- Overall accuracy comparison
- Confusion matrices (heatmaps)

Classes (Labels):
-----------------
0 - Motorbikes
1 - School Bus
2 - Touring Bike
3 - Airplanes
4 - Car Side

"""


import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------
# Load Training Data
# ----------------------------
print("[INFO] Loading training data...")
X_train = np.load("index.npy").astype(np.float32)
img_paths = np.load("paths.npy")

def get_label(path):
    if 'motorbikes' in path:
        return 0
    elif 'school' in path:
        return 1
    elif 'touring' in path:
        return 2
    elif 'airplanes' in path:
        return 3
    else:
        return 4

y_train = np.array([get_label(p) for p in img_paths], dtype=np.int32)

# ----------------------------
# Load Vocabulary & Setup Extractor
# ----------------------------
sift = cv.SIFT_create()
vocabulary = np.load("vocabulary.npy")
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

# ----------------------------
# Load Test Data
# ----------------------------
print("[INFO] Extracting test data...")
test_folders = [
    '145.motorbikes-101_test',
    '178.school-bus_test',
    '224.touring-bike_test',
    '251.airplanes-101_test',
    '252.car-side-101_test'
]

X_test, y_test = [], []

for folder in test_folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)
        if bow_desc is not None:
            X_test.append(bow_desc.flatten())
            y_test.append(get_label(path))

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

# ----------------------------
# SVM Prediction
# ----------------------------
print("[INFO] Loading and running SVM...")
svm = cv.ml.SVM_create()
svm = svm.load('svm')

y_pred_svm = []
for desc in X_test:
    _, pred = svm.predict(desc.reshape(1, -1))
    y_pred_svm.append(int(pred[0][0]))

# ----------------------------
# k-NN Prediction
# ----------------------------
print("[INFO] Training and running k-NN...")
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
print("\n[RESULT] Evaluation Metrics\n")

print("--- SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

print("--- k-NN ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ----------------------------
# Confusion Matrix Plot
# ----------------------------
labels = ["Motorbikes", "School Bus", "Touring Bike", "Airplanes", "Car Side"]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axs[0])
axs[0].set_title('SVM Confusion Matrix')
axs[0].set_xlabel('Predicted')
axs[0].set_ylabel('True')

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=axs[1])
axs[1].set_title('k-NN Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')

plt.tight_layout()
plt.show()
