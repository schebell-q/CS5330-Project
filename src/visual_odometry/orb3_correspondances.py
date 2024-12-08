import cv2
import numpy as np
from typing import Tuple
from visual_odometry.correspondances import Correspondances

def find_correspondance_orb3(im1: np.array, im2: np.array, K: np.ndarray, max_features: int, feature_distance_threshold: float) -> Correspondances:
    F, E, p1, p2 = get_matches_ORB(im1, im2, K, max_features, feature_distance_threshold, fmat=True)
    return Correspondances(im1, im2, p1, p2)

def get_matches_ORB(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    max_features,
    feature_distance_threshold,
    fmat: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect and match ORB features between two images, estimate the Fundamental and Essential matrices,
    and return the inlier matched points.

    Args:
        img1 (np.ndarray): The first image (grayscale or color).
        img2 (np.ndarray): The second image (grayscale or color).
        K (np.ndarray): The camera intrinsic matrix (3x3).
        fmat (bool): Flag indicating whether to compute the Fundamental matrix. If False, only match features.

    Returns:
        F (np.ndarray): The estimated Fundamental matrix (3x3) if fmat is True, else None.
        E (np.ndarray): The estimated Essential matrix (3x3) if fmat is True, else None.
        inliers_a (np.ndarray): Inlier points from the first image (Nx2).
        inliers_b (np.ndarray): Inlier points from the second image (Nx2).
    """
    # Convert images to grayscale if they are in color
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=4000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    if des1 is None or des2 is None:
        print("Warning: No descriptors found in one or both images.")
        return None, None, None, None
    
    # Initialize BFMatcher with Hamming distance and crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Perform k-NN matching with k=2
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    pts_a = []
    pts_b = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts_a.append(kp1[m.queryIdx].pt)
            pts_b.append(kp2[m.trainIdx].pt)
    
    pts_a = np.asarray(pts_a)
    pts_b = np.asarray(pts_b)
    
    if len(good_matches) < 8:
        print("Warning: Not enough good matches found.")
        return None, None, None, None
    
    if fmat:
        # Estimate Fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None:
            print("Warning: Fundamental matrix estimation failed.")
            return None, None, None, None
        
        # Select inlier matches
        inliers_a = pts_a[mask.ravel() == 1]
        inliers_b = pts_b[mask.ravel() == 1]
        
        # Compute Essential matrix
        E = K.T @ F @ K
        
        return F, E, inliers_a, inliers_b
    else:
        # If Fundamental matrix is not needed, return matches without estimation
        return None, None, pts_a, pts_b

