import cv2
import numpy as np
import torch

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline

from visual_odometry.correspondances import Correspondances


def find_correspondance_gluestick(im1: np.array, im2: np.array, max_pts: int, max_lines: int) -> Correspondances:
    # This is a high level function that converts GlueStick output into the right data type
    kp1, kp2 = run_gluestick(im1, im2, max_pts, max_lines)
    return Correspondances(im1, im2, kp1, kp2)


def run_gluestick(im1: np.array, im2: np.array, max_pts: int, max_lines: int) -> (np.array, np.array):
    # The guts of using GlueStick should go here
    # This function can be imported in a notebook and run directly

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    gray0 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    pred = pipeline_model(x)

    pred = batch_to_np(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
    line_matches = pred["line_matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    valid_matches = line_matches != -1
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]

    return matched_kps0, matched_kps1
