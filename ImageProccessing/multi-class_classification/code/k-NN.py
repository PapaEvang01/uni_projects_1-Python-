"""
====================================================================
Multi-Class Image Classification with Custom k-NN using BOW & OpenCV
====================================================================

Description:
------------
This script implements a custom k-Nearest Neighbors (k-NN) classifier
for multi-class image classification using a Bag-of-Visual-Words (BOW)
representation extracted with SIFT features. The classification is done
manually without using OpenCV's built-in k-NN module.

Classes:
- Motorbikes
- School Bus
- Touring Bike
- Airplanes
- Car Side

Inputs:
-------
- 'vocabulary.npy': Visual words vocabulary (KMeans clusters)
- 'index.npy': BOW descriptors for training images
- 'paths.npy': File paths of training images
- Test image folders for each class (e.g. '145.motorbikes-101_test')

Methodology:
------------
1. Load vocabulary and BOW descriptors.
2. Manually implement k-NN classification based on Euclidean distance.
3. Evaluate accuracy on a separate test set.
4. Perform k-fold cross-validation to analyze the effect of different k.

Output:
-------
- Terminal output showing predictions and accuracy per class
- A plot showing accuracy vs number of neighbors (k)

"""
import cv2 as cv
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Load Precomputed BOW and Vocabulary
# ============================================
sift = cv.SIFT_create()

# Load saved vocabulary, BOW descriptors, and image paths
vocabulary = np.load('vocabulary.npy')
bow_descs = np.load('index.npy').astype(np.float32)
img_paths = np.load('paths.npy')

# Set up descriptor extractor with the loaded vocabulary
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

# ============================================
# STEP 2: Assign Class Labels to Training Images
# ============================================
labels = []
for path in img_paths:
    if 'motorbikes' in path:
        labels.append(1)
    elif 'school' in path:
        labels.append(2)
    elif 'touring' in path:
        labels.append(3)
    elif 'airplanes' in path:
        labels.append(4)
    else:
        labels.append(5)

labels = np.array(labels, dtype=np.int32)

# ============================================
# STEP 3: Load and Label Test Images
# ============================================
test_folders = [
    '145.motorbikes-101_test',
    '178.school-bus_test',
    '224.touring-bike_test',
    '251.airplanes-101_test',
    '252.car-side-101_test'
]

labels_test = []
test_images = []

for folder in test_folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        image = cv.imread(path)
        test_images.append((image, path))

        # Assign label
        if 'motorbikes' in path:
            labels_test.append(1)
        elif 'school' in path:
            labels_test.append(2)
        elif 'touring' in path:
            labels_test.append(3)
        elif 'airplanes' in path:
            labels_test.append(4)
        else:
            labels_test.append(5)

labels_test = np.array(labels_test, dtype=np.int32)

# ============================================
# STEP 4: Define Custom k-NN Classifier
# ============================================
def classify_knn(img, k=25):
    keypoints = sift.detect(img)
    desc = descriptor_extractor.compute(img, keypoints)
    if desc is None:
        return 0  # Unknown class

    # Compute Euclidean distances to all training descriptors
    distances = [np.linalg.norm(desc - d) for d in bow_descs]
    sorted_indices = np.argsort(distances)[:k]
    top_labels = labels[sorted_indices]

    # Majority vote
    predicted_label = np.bincount(top_labels).argmax()
    return predicted_label

# ============================================
# STEP 5: Test All Images and Collect Predictions
# ============================================
predicted_labels = []

for img, path in test_images:
    print("\n--------------------------------------")
    print(f"Classifying: {path}")
    predicted = classify_knn(img, k=25)
    predicted_labels.append(predicted)

# ============================================
# STEP 6: Evaluation and Accuracy Reporting
# ============================================
print("\n[RESULT] Classification Report")
overall_accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Accuracy per class
for class_id, class_name in zip(range(1, 6), [
    "Motorbikes", "School Bus", "Touring Bike", "Airplanes", "Car Side"
]):
    idx = np.where(labels_test == class_id)[0]
    acc = accuracy_score(labels_test[idx], np.array(predicted_labels)[idx])
    print(f"{class_name}: {acc:.4f}")

# ============================================
# STEP 7: Plot Accuracy vs k using Cross-Validation
# ============================================
print("\n[INFO] Performing cross-validation to find optimal k...")
k_range = range(1, 31)
cv_scores = []

model = KNeighborsClassifier()
for k in k_range:
    model.n_neighbors = k
    score = cross_val_score(model, bow_descs, labels, cv=10, scoring='accuracy').mean()
    cv_scores.append(score)

plt.figure()
plt.plot(k_range, cv_scores, marker='o')
plt.title('Accuracy vs k (10-fold Cross-Validation)')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.grid(True)
plt.show()
