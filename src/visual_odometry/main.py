import argparse
import itertools
import logging
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from visual_odometry.plotting import plot_poses
from .geometry import convert_correspondance_to_transform, PoseDelta
from .sift_correspondances import find_correspondance_sift
from .gluestick_correspondances import find_correspondance_gluestick

logger = logging.getLogger(__name__)


class PoseEstimator(object):
    """An iterable estimator for changes in pose. Operates on an iterable that generates images,
    including streams of images."""
    num_frames_so_far: int = None
    feature_detector: str = None
    max_num_features: int = None
    feature_distance_threshold: float = None
    ransac_prob_success: float = None
    camera_intrinsic_matrix: np.array = None
    image_generator: Iterator[np.array] = None
    last_image: np.array = None

    def __init__(self, image_generator: Iterable[np.array], feature_detector: str, max_num_features: int,
                 feature_distance_threshold: float, ransac_prob_success: float):
        self.feature_detector = feature_detector
        self.max_num_features = max_num_features
        self.feature_distance_threshold = feature_distance_threshold
        self.ransac_prob_success = ransac_prob_success
        self.image_generator = iter(image_generator)

        self.last_image = next(self.image_generator)
        self.num_frames_so_far = 1

        # TODO calibrate camera matrix
        # This is from PA5
        fx = 1392.1069298937407
        px = 980.1759848618066
        py = 604.3534182680304
        self.camera_intrinsic_matrix = np.array([[fx, 0, px], [0, fx, py], [0, 0, 1]])

    def __iter__(self):
        return self

    def __next__(self) -> PoseDelta:
        new_image = next(self.image_generator)
        self.num_frames_so_far += 1
        logger.info(f'Processing frame {self.num_frames_so_far}')

        # Compute correspondance between images
        if self.feature_detector == 'sift':
            correspondance = find_correspondance_sift(self.last_image, new_image, self.max_num_features,
                                                      self.feature_distance_threshold)
        elif self.feature_detector == 'gluestick':
            correspondance = find_correspondance_gluestick(self.last_image, new_image, self.max_num_features,
                                                           self.max_num_features // 3)
        else:
            raise ValueError(f'unrecognized feature detector type "{self.feature_detector}"')

        # Find transform from correspondance
        pose_delta = convert_correspondance_to_transform(correspondance, self.camera_intrinsic_matrix,
                                                         self.ransac_prob_success)

        # Save data that will be used for computing next delta
        self.last_image = new_image

        return pose_delta


def read_images(filename: Path) -> Iterable[np.array]:
    if filename.is_dir():
        logger.info(f'Reading files in {filename}')
        # read all video/images in dir
        return itertools.chain(*map(read_images, sorted(filename.iterdir())))

    ext = filename.suffix.lower()
    if ext in ['.png', '.jpeg', '.jpg']:
        logger.info(f'Reading image {filename}')
        img = cv2.imread(str(filename.absolute()))
        return [img]
    if ext in ['.mp4', '.mov']:
        imgs = []
        logger.info(f'Reading video {filename}')
        cap = cv2.VideoCapture(str(filename.absolute()))
        while cap.isOpened():
            status, frame = cap.read()
            if status:
                imgs.append(frame)
            else:
                break
        cap.release()
        logger.info(f'Read {len(imgs)} frames')
        return imgs
    raise TypeError(f'unrecognized file type "{ext}"')


def main():
    parser = argparse.ArgumentParser(
        prog='vo',
        description='Converts input images to an estimated trajectory and graphs',
    )
    parser.add_argument('-f', '--feature-detector', nargs='?', default='sift',
                        choices=['sift', 'gluestick'],
                        help='the feature detector algorithm to use to identify keypoints')
    parser.add_argument('-d', '--feature-distance', nargs='?', type=float,
                        default=0.8,  # default number from pa5 vo.py
                        help='the maximum feature-space distance between two features to be considered a match')
    parser.add_argument('--max-features', nargs='?', type=int,
                        default=4000,  # default number from pa5 vo.py
                        help='the maximum number of features to use from the feature detector')
    parser.add_argument('--max-images', nargs='?', type=int, default=0,
                        help='the maximum number of features to use from the feature detector')
    parser.add_argument('-r', '--ransac-prob-success', nargs='?', type=float, default=0.9,
                        help='probability of success desired for RANSAC matching. higher takes longer to run.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='set to include all debug output')
    parser.add_argument('-o', '--output', nargs='?', type=Path, default=Path('trajectory.png'),
                        help='path to save trajectory plot')
    parser.add_argument('filename', nargs='+', type=Path,
                        help='path to a video or image, or directory of videos or images')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # read in videos or images
    images = list(itertools.chain(*map(read_images, args.filename)))
    if args.max_images > 0:
        images = images[:args.max_images]

    pose_deltas = list(PoseEstimator(images, args.feature_detector, args.max_features, args.feature_distance,
                                     args.ransac_prob_success))

    trajectory = list(itertools.accumulate(pose_deltas, initial=PoseDelta(np.eye(4))))

    plot_poses(trajectory, args.output, figsize=(7, 8))


if __name__ == "__main__":
    main()
