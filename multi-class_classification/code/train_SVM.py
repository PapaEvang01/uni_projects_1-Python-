"""
===============================================================
 Multi-Class Image Classification using BOW and OpenCV SVM
===============================================================

Description:
------------
This script trains a multi-class Support Vector Machine (SVM)
classifier using OpenCV's ML module. It follows the Bag-of-Visual-Words
(BOW) model and SIFT feature extraction to classify images into 5 object classes.

Classes:
- Motorbikes
- School Bus
- Touring Bike
- Airplanes
- Car Side

Inputs:
-------
- 'index.npy': Numpy array of BOW descriptors for all training images
- 'paths.npy': List of file paths to the training images (used for label extraction)

Method:
-------
1. Load BOW features and image paths.
2. Assign integer labels to each image based on folder name.
3. Configure and train an OpenCV SVM with RBF kernel.
4. Save the trained model to disk.

Output:
-------
- 'svm': Trained OpenCV SVM model (can be loaded later for prediction)

"""
import os
import cv2 as cv
import numpy as np

# ========================================
# SVM TRAINING ON BOW DESCRIPTORS (OpenCV)
# ========================================

# --- Step 1: Define Training Class Folders ---
train_folders = [
    '145.motorbikes-101',
    '178.school-bus',
    '224.touring-bike',
    '251.airplanes-101',
    '252.car-side-101'
]

# --- Step 2: Load Precomputed BOW Descriptors & Paths ---
print('[INFO] Loading BOW descriptors and image paths...')
bow_descs = np.load('index.npy').astype(np.float32)  # Features
img_paths = np.load('paths.npy')                     # Corresponding file paths

# --- Step 3: Assign Integer Labels per Class ---
print('[INFO] Assigning labels...')
labels = []
for path in img_paths:
    if 'motorbikes' in path:
        labels.append(0)
    elif 'school' in path:
        labels.append(1)
    elif 'touring' in path:
        labels.append(2)
    elif 'airplanes' in path:
        labels.append(3)
    else:
        labels.append(4)

labels = np.array(labels, dtype=np.int32)

# --- Step 4: Initialize and Configure SVM ---
print('[INFO] Training SVM model...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)              # C-Support Vector Classification
svm.setKernel(cv.ml.SVM_RBF)              # Radial Basis Function kernel
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1e-6))  # Stop after 100 iterations or epsilon

# Optional: Set class weights (all equal here, can be tuned if dataset is imbalanced)
svm.setClassWeights(np.ones(5, dtype=np.float32))

# --- Step 5: Train the SVM on the Training Data ---
svm.train(bow_descs, cv.ml.ROW_SAMPLE, labels)

# --- Step 6: Save the Trained Model to File ---
svm.save('svm')  # Saves to file named "svm" (no extension by default)

print('[INFO] SVM training complete. Model saved to file.')
