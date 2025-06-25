"""
===============================================================
 Multi-Class Image Classification using BOVW + SIFT (Step 1)
===============================================================

 Description:
This script prepares the visual vocabulary and index descriptors for a multi-class image classification system based on the Bag of Visual Words (BOVW) model using OpenCV and SIFT.

 Input:
- A subset of the Caltech-256 dataset organized in 5 class folders:
  ['145.motorbikes-101', '178.school-bus', '224.touring-bike',
   '251.airplanes-101', '252.car-side-101']
- Each folder contains images for training.

 Output:
- 'vocabulary.npy': the K-means visual vocabulary (centroids)
- 'index.npy': the BOW descriptors for each training image
- 'paths.npy': the corresponding image file paths

️ Methodology:
1. Extract local SIFT features (keypoints + descriptors) from all training images.
2. Cluster all descriptors using K-means to build a visual vocabulary.
3. Compute a global descriptor (histogram of visual words) for each image.
4. Save the descriptor index and vocabulary for use in later classification steps.
"""

import os
import cv2 as cv
import numpy as np

# --------------------------------------------
#  STEP 1: Setup
# --------------------------------------------

# List of training folders (one per class)
train_folders = [
    '145.motorbikes-101',
    '178.school-bus',
    '224.touring-bike',
    '251.airplanes-101',
    '252.car-side-101'
]

# Create a SIFT detector
sift = cv.SIFT_create()

# --------------------------------------------
# STEP 2: Feature Extraction
# --------------------------------------------

def extract_local_features(image_path):
    """Extract SIFT descriptors from an image"""
    img = cv.imread(image_path)
    if img is None:
        return None
    keypoints = sift.detect(img, None)
    _, descriptors = sift.compute(img, keypoints)
    return descriptors

# Extract all descriptors from training images
print('[INFO] Extracting SIFT descriptors from training set...')
all_descriptors = np.zeros((0, 128), dtype=np.float32)

for folder in train_folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        desc = extract_local_features(path)
        if desc is not None:
            all_descriptors = np.vstack((all_descriptors, desc))

# --------------------------------------------
#  STEP 3: Create Visual Vocabulary (KMeans)
# --------------------------------------------

print('[INFO] Creating visual vocabulary using KMeans...')
term_criteria = (cv.TERM_CRITERIA_EPS, 30, 0.1)
num_clusters = 50  # Size of visual vocabulary
trainer = cv.BOWKMeansTrainer(num_clusters, term_criteria, 1, cv.KMEANS_PP_CENTERS)

vocabulary = trainer.cluster(all_descriptors)
np.save('vocabulary.npy', vocabulary)

# --------------------------------------------
#  STEP 4: Compute BOW descriptors for all images
# --------------------------------------------

print('[INFO] Building image descriptor index...')
# Initialize BOW descriptor extractor with SIFT + Brute Force Matcher
bow_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
bow_extractor.setVocabulary(vocabulary)

# Store paths and their corresponding BOW vectors
img_paths = []
bow_descriptors = np.zeros((0, vocabulary.shape[0]), dtype=np.float32)

for folder in train_folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv.imread(path)
        if img is None:
            continue
        kp = sift.detect(img, None)
        bow_desc = bow_extractor.compute(img, kp)
        if bow_desc is not None:
            img_paths.append(path)
            bow_descriptors = np.vstack((bow_descriptors, bow_desc))

# --------------------------------------------
#  STEP 5: Save Results
# --------------------------------------------

np.save('index.npy', bow_descriptors)
np.save('paths.npy', img_paths)

print('[DONE] Visual vocabulary and index saved successfully.')
