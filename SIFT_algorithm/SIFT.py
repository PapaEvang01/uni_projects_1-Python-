import numpy as np
import cv2 as cv

# =====================================================
# 🧵 Panorama Stitching using SIFT and Manual Cross-Check
# =====================================================
# This script stitches four images into a panoramic view using:
# 1. SIFT keypoint detection
# 2. Custom brute-force matching with cross-check
# 3. Homography estimation (RANSAC)
# 4. Warping and cropping to generate seamless transitions

# --- Matching Function (One-Way using Euclidean Distance) ---
def match_descriptors(desc1, desc2):
    matches = []
    for i in range(desc1.shape[0]):
        fv = desc1[i, :]
        distances = np.linalg.norm(desc2 - fv, axis=1)
        j = np.argmin(distances)
        matches.append(cv.DMatch(i, j, distances[j]))
    return matches

# --- Cross-Check Matching (Keep only mutually best matches) ---
def cross_check(desc1, desc2):
    matches_12 = match_descriptors(desc1, desc2)
    matches_21 = match_descriptors(desc2, desc1)

    good = []
    for m in matches_12:
        forward_q = m.queryIdx
        forward_t = m.trainIdx

        for n in matches_21:
            if n.trainIdx == forward_q and n.queryIdx == forward_t:
                good.append(m)
                break

    return list(dict.fromkeys(good))

# --- Homography + Warping + Cropping ---
def stitch_pair(img1, img2, kp1, desc1, kp2, desc2, title='Panorama'):
    good = cross_check(desc1, desc2)
    print(f"{title}: {len(good)} good matches")

    if len(good) < 4:
        print(f"[ERROR] Not enough matches for {title}. Skipping.")
        return img1

    # Extract matched point coordinates
    img_pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    img_pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, _ = cv.findHomography(img_pts2, img_pts1, cv.RANSAC)

    # Warp second image onto first image's space
    panorama = cv.warpPerspective(img2, H, (img1.shape[1] + 1000, img1.shape[0] + 1000))
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1

    # Crop black borders using thresholding and contour bounding
    gray = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    x, y, w, h = cv.boundingRect(contours[0])
    result = panorama[y:y+h-20, x:x+w-20]

    cv.imshow(f'Stitched Pair - {title}', result)
    cv.waitKey(0)
    return result

# --- Main Function: Load Images, Detect, Match and Stitch ---
def build_panorama():
    sift = cv.SIFT_create()

    # Load and resize images (800x600 for uniformity)
    img1 = cv.resize(cv.imread('rio-01.png'), (800, 600))
    img2 = cv.resize(cv.imread('rio-02.png'), (800, 600))
    img3 = cv.resize(cv.imread('rio-03.png'), (800, 600))
    img4 = cv.resize(cv.imread('rio-04.png'), (800, 600))

    # Display original images
    cv.imshow('Image 1 - rio-01', img1)
    cv.imshow('Image 2 - rio-02', img2)
    cv.imshow('Image 3 - rio-03', img3)
    cv.imshow('Image 4 - rio-04', img4)
    cv.waitKey(0)

    # Compute SIFT keypoints and descriptors
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    kp3, desc3 = sift.detectAndCompute(img3, None)
    kp4, desc4 = sift.detectAndCompute(img4, None)

    # Optional: Visualize keypoints for debugging
    cv.imshow('Keypoints 1', cv.drawKeypoints(img1, kp1, None))
    cv.imshow('Keypoints 2', cv.drawKeypoints(img2, kp2, None))
    cv.imshow('Keypoints 3', cv.drawKeypoints(img3, kp3, None))
    cv.imshow('Keypoints 4', cv.drawKeypoints(img4, kp4, None))
    cv.waitKey(0)

    # First-level stitching: (img1 + img2), (img3 + img4)
    crop12 = stitch_pair(img1, img2, kp1, desc1, kp2, desc2, title='Pair 1-2')
    crop34 = stitch_pair(img3, img4, kp3, desc3, kp4, desc4, title='Pair 3-4')

    # Second-level stitching: merge crop12 with crop34
    kp12, desc12 = sift.detectAndCompute(crop12, None)
    kp34, desc34 = sift.detectAndCompute(crop34, None)
    final_pano = stitch_pair(crop12, crop34, kp12, desc12, kp34, desc34, title='Final')

    # Show and save final panorama
    cv.imshow('Final Panorama (SIFT)', final_pano)
    cv.imwrite('panorama_sift_final.png', final_pano)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    build_panorama()
