"""
===============================================================
 Cell Detection and Analysis from Noisy Microscopy Images 
===============================================================

This Python script performs automatic detection and analysis of cells
in grayscale images that have been corrupted by salt-and-pepper noise.

🔍 The pipeline includes:
- Random selection of matching 'original' and 'noisy' image pairs
- Manual implementation of a median filter (3x3) for denoising
- Morphological opening to remove small noise artifacts
- Thresholding (binary segmentation)
- Connected component analysis to detect individual cells
- Bounding box annotation for each detected cell
- Calculation of:
    • Cell area (pixel count)
    • Bounding box area
    • Mean grayscale intensity (via manually computed integral image)

🖼️ The results are displayed with bounding boxes and printed stats.

📂 Folder Structure:
- original/ → folder with clean original images
- noise/    → folder with salt-and-pepper noisy versions (same filenames)

 Dependencies:
- OpenCV (cv2)
- NumPy
- Random
- OS

Coursework: [e.g., Digital Image Processing - University Project]

"""

import cv2
import numpy as np
import os
import random

# --- Random image selector ---
def choose_random_image():
    files = [f for f in os.listdir('original') if f.endswith('.png') or f.endswith('.jpg')]
    chosen = random.choice(files)
    return os.path.join('original', chosen), os.path.join('noise', chosen), chosen

# --- Manual Median Filter ---
def manual_median_filter(img):
    m, n = img.shape
    filtered = img.copy()
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            window = [
                img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                img[i, j - 1], img[i, j], img[i, j + 1],
                img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]
            ]
            window.sort()
            filtered[i, j] = window[4]
    return filtered.astype(np.uint8)

# --- Integral Image (without OpenCV) ---
def compute_integral_image(gray):
    m, n = gray.shape
    integral = np.zeros((m + 1, n + 1), dtype=np.float64)

    for i in range(m):
        row_sum = 0
        for j in range(n):
            row_sum += gray[i, j]
            integral[i + 1, j + 1] = integral[i, j + 1] + row_sum
    return integral

# --- Region Analysis & Bounding Box Annotation ---
def analyze_regions(gray, binary, output_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    integral = compute_integral_image(gray)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    region = 0

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        bound_area = w * h
        region += 1

        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output_img, str(region), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        sum_pixels = integral[y, x] + integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x]
        avg_intensity = sum_pixels / bound_area

        print(f"Region #{region}")
        print(f" - Area (px): {area}")
        print(f" - Bounding Box Area: {bound_area}")
        print(f" - Mean Intensity in Box: {avg_intensity:.2f}")

    return output_img

# --- Main ---
def main():
    original_path, noisy_path, filename = choose_random_image()
    print(f"Selected image: {filename}")

    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

    filtered = manual_median_filter(noisy_img)

    # Morphological opening
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, strel)

    # Thresholding
    _, binary = cv2.threshold(opened, 20, 255, cv2.THRESH_BINARY)

    # Analysis
    result = analyze_regions(opened, binary, original_img)

    # Show results
    cv2.imshow("Filtered", filtered)
    cv2.imshow("Opened", opened)
    cv2.imshow("Binary", binary)
    cv2.imshow("Final Result with Bounding Boxes", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
