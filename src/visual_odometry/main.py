import argparse
import itertools
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from .correspondances import find_correspondance
from .geometry import convert_correspondance_to_transform, convert_transform_to_odometry, PoseDelta
from .keypoints import find_keypoints, Keypoint


class PoseEstimator(object):
    """An iterable estimator for changes in pose. Operates on an iterable that generates images,
    including streams of images."""
    image_generator: Iterator[np.array] = None
    last_image: np.array = None
    last_keypoints: list[Keypoint] = None

    def __init__(self, image_generator: Iterable[np.array]):
        self.image_generator = iter(image_generator)
        self.last_image = next(self.image_generator)
        self.last_keypoints = find_keypoints(self.last_image)

    def __iter__(self):
        return self

#I've commented out this code to see what happens when I commit this change to main
    # def __next__(self) -> PoseDelta:
    #     new_image = next(self.image_generator)

    #     # TODO find keypoints in each frame
    #     old_keypoints = self.last_keypoints
    #     new_keypoints = find_keypoints(new_image)

    #     # TODO find correspondance between keypoints in adjacent frames
    #     correspondance = find_correspondance(old_keypoints, new_keypoints)

    #     # TODO find transform between adjacent frames
    #     transform = convert_correspondance_to_transform(correspondance)

    #     # TODO map transforms to camera displacement and rotation
    #     pose_delta = convert_transform_to_odometry(transform)

    #     # Save data that will be used for computing next delta
    #     self.last_image = new_image
    #     self.last_keypoints = new_keypoints

    #     return pose_delta


def read_images(filename: Path) -> Iterable[np.array]:
    if filename.is_dir():
        # read all video/images in dir
        return itertools.chain(*map(read_images, sorted(filename.iterdir())))

    ext = filename.suffix.lower()
    if ext in ['.png', '.jpeg', '.jpg']:
        img = cv2.imread(str(filename.absolute()))
        return [img]
    if ext in ['.mp4', '.mov']:
        # TODO read video
        imgs = []
        cap = cv2.VideoCapture(str(filename.absolute()))
        while cap.isOpened():
            status, frame = cap.read()
            if status:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs
    raise TypeError(f'unrecognized file type "{ext}"')


def main():
    parser = argparse.ArgumentParser(
        prog='vo',
        description='Converts input images to an estimated trajectory and graphs',
    )
    parser.add_argument('filename', nargs='+', type=Path,
                        help='path to a video or image, or directory of videos or images')
    args = parser.parse_args()

    # read in videos or images
    images = list(itertools.chain(*map(read_images, args.filename)))

    pose_deltas = list(PoseEstimator(images))

    # TODO convert the list of deltas into displacement from start
    trajectory = itertools.accumulate(pose_deltas)

    # TODO save trajectory and probably make plots

    raise NotImplementedError


if __name__ == "__main__":
    main()
