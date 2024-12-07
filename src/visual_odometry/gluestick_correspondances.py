import numpy as np

from visual_odometry.correspondances import Correspondances


def find_correspondance_gluestick(im1: np.array, im2: np.array, other_config_args) -> Correspondances:
    # This is a high level function that converts GlueStick output into the right data type
    kp1, kp2 = run_gluestick(im1, im2, other_config_args)
    return Correspondances(im1, im2, kp1, kp2)


def run_gluestick(im1: np.array, im2: np.array, other_config_args) -> (np.array, np.array):
    # The guts of using GlueStick should go here
    # This function can be imported in a notebook and run directly
    raise NotImplementedError()
