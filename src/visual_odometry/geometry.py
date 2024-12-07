import cv2
from dataclasses import dataclass
import logging
import numpy as np


from .correspondances import Correspondances

logger = logging.getLogger(__name__)


@dataclass
class PoseDelta:
    se3_transform: np.array

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'can only add {self.__class__} to other {self.__class__}')

        return PoseDelta(self.se3_transform @ other.se3_transform)


def get_emat_from_fmat(F: np.array, K1: np.array, K2: np.array) -> np.array:
    """Adapted from PA5 (Quinn Schebell).

    Create essential matrix from camera instrinsics and fundamental matrix
    Args:
        F:  A numpy array of shape (3, 3) representing the fundamental matrix between
            two cameras
        K1: A numpy array of shape (3, 3) representing the intrinsic matrix of the
            first camera
        K2: A numpy array of shape (3, 3) representing the intrinsic matrix of the
            second camera

    Returns:
        E:  A numpy array of shape (3, 3) representing the essential matrix between
            the two cameras.
    """

    E = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    E = K2.T @ F @ K1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return E


def convert_correspondance_to_transform(correspondances: Correspondances, K: np.array,
                                        ransac_prob_success: float) -> PoseDelta:
    # Adapted from PA5 vo.py.
    F, matched_points_a, matched_points_b = ransac_fundamental_matrix(correspondances.p1, correspondances.p2,
                                                                      ransac_prob_success)
    E = get_emat_from_fmat(F, K1=K, K2=K)
    _num_inlier, i2Ri1, i2ti1, _ = cv2.recoverPose(E, matched_points_a, matched_points_b)
    i2Ti1 = np.eye(4)
    i2Ti1[:3, :3] = i2Ri1
    i2Ti1[:3, 3] = i2ti1.squeeze()

    # convert coordinate frame 1 into coordinate frame 1
    # assume 1 meter translation for unknown scale (gauge ambiguity)
    i1Ti2 = np.linalg.inv(i2Ti1)
    return PoseDelta(i1Ti2)


def calculate_num_ransac_iterations(prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
    """Adapted from PA5 (Quinn Schebell).

    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: [float] representing the desired guarantee of success
        sample_size: [int] the number of samples included in each RANSAC
            iteration
        ind_prob_correct: [float] representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    prob_no_bad = ind_prob_correct ** sample_size
    prob_some_bad = 1 - prob_no_bad
    # prob_some_bad ** num_samples = prob_all_trials_bad
    # 1 - prob_all_trials_bad = prob_some_trial_good = prob_success
    # prob_success = 1 - prob_some_bad ** num_samples
    # prob_some_bad ** num_samples = 1 - prob_success
    # num_samples * log(prob_some_bad) = log(1 - prob_success)
    num_samples = np.log(1 - prob_success) / np.log(prob_some_bad)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def normalize_points(points: np.array) -> (np.array, np.array):
    """Adapted from PA5 (Quinn Schebell).

    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 3) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 3) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """

    points_normalized, T = None, None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    N, _ = points.shape
    # (N, 2)
    points_2d = (points.T[:2] / points.T[2]).T
    # (2,)
    means = np.mean(points_2d, axis=0)
    # (N, 2)
    points_zero_centered = points_2d - means
    # (1,)
    mean_dist = np.mean(np.linalg.norm(points_zero_centered, axis=1))

    scale = 2 ** 0.5 / mean_dist

    # (3, 4)
    T = np.array([
        [scale, 0, -means[0] * scale],
        [0, scale, -means[1] * scale],
        [0, 0, 1],
    ])

    # (N, 3)
    points_normalized = points @ T.T

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.array, T_a: np.array, T_b: np.array) -> np.array:
    """Adapted from PA5 (Quinn Schebell).

    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """

    F_orig = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T @ F_norm @ T_a

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(points_a: np.array, points_b: np.array) -> np.array:
    """Adapted from PA5 (Quinn Schebell).

    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    F = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    N, _ = points_a.shape
    a_hat, T_a = normalize_points(np.append(points_a, np.ones((N, 1)), axis=1))
    b_hat, T_b = normalize_points(np.append(points_b, np.ones((N, 1)), axis=1))

    u = a_hat[:, 0]
    v = a_hat[:, 1]
    up = b_hat[:, 0]
    vp = b_hat[:, 1]
    # uu'F11 + vu'F12 + u'F13 +
    # uv'F21 + vv'F22 + v'F23 +
    # u  F31 + v  F32 +   F33
    # = 0
    eqns = np.zeros((N, 9))
    eqns[:, 0] = u * up
    eqns[:, 1] = v * up
    eqns[:, 2] = up
    eqns[:, 3] = u * vp
    eqns[:, 4] = v * vp
    eqns[:, 5] = vp
    eqns[:, 6] = u
    eqns[:, 7] = v
    eqns[:, 8] = 1

    eigval, eigvec = np.linalg.eig(eqns.T @ eqns)
    F_norm = eigvec[:, np.argmin(eigval)].reshape((3, 3))

    U, D, V = np.linalg.svd(F_norm)
    D[-1] = 0
    F_rank2 = U @ np.diag(D) @ V

    F = unnormalize_F(F_rank2, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F


ransac_print_period = 1000

def ransac_fundamental_matrix(matches_a: np.array, matches_b: np.array,
                              ransac_prob_success: float) -> (np.array, np.array, np.array):
    """Adapted from PA5 (Quinn Schebell).

    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers.
        3. Consider the geometric distances of a keypoint to its estimated
           epipolar line. It is a bit more robust and the error threshold
           is easier to interpret (why?). Check the slide #70 in
           cs5330-fall-2022-18.pptx.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F


    """

    best_F, inliers_a, inliers_b = None, None, None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    N, _ = matches_a.shape
    num_samples = calculate_num_ransac_iterations(ransac_prob_success, 4, 0.2)
    threshold = 1e-5
    max_matches = 30

    logger.info(f'Running RANSAC for {num_samples} iterations')

    a_h = np.append(matches_a, np.ones((N, 1)), axis=1)
    b_h = np.append(matches_b, np.ones((N, 1)), axis=1)

    best_score = 0
    best_F = None
    inliers_a = None
    inliers_b = None
    for i in range(num_samples):
        if i % ransac_print_period == 0:
            logger.debug(f'Iteration {i}/{num_samples}')

        samples = np.random.choice(np.arange(N), (8,), replace=False)
        sample_a = matches_a[samples]
        sample_b = matches_b[samples]
        F = estimate_fundamental_matrix(sample_a, sample_b)

        dists = np.abs(np.diag(b_h @ F @ a_h.T))
        inliers = dists < threshold
        score = np.sum(inliers)

        if score > best_score:
            best_score = score
            best_F = F
            sub_inliers = np.argsort(dists)[:min(score, max_matches)]
            inliers_a = matches_a[sub_inliers]
            inliers_b = matches_b[sub_inliers]

    logger.debug(f'RANSAC completed with {len(inliers_a)} matches')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
