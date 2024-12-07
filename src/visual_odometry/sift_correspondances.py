import cv2
import numpy as np

from visual_odometry.correspondances import Correspondances


def find_correspondance_sift(im1: np.array, im2: np.array, max_features: int, feature_distance_threshold: float) -> Correspondances:
    p1, p2 = get_matches(im1, im2, max_features, feature_distance_threshold)
    return Correspondances(im1, im2, p1, p2)


def get_matches(pic_a: np.array, pic_b: np.array, n_feat: int, distance_threshold: float) -> (np.array, np.array):
    """Copied from PA5 utils.py.

    Get unreliable matching points between two images using SIFT.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        pic_a: a numpy array representing image 1.
        pic_b: a numpy array representing image 2.
        n_feat: an int representing number of matching points required.

    Returns:
        pts_a: a numpy array representing image 1 points.
        pts_b: a numpy array representing image 2 points.
    """
    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.SIFT_create()

    kp_a, desc_a = sift.detectAndCompute(pic_a, None)
    kp_b, desc_b = sift.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_L2)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance / n.distance <= distance_threshold:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches[: int(n_feat)]:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)
