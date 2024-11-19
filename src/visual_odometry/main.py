from itertools import accumulate
from typing import Iterable, Iterator

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

    def __next__(self) -> PoseDelta:
        new_image = next(self.image_generator)

        # TODO find keypoints in each frame
        old_keypoints = self.last_keypoints
        new_keypoints = find_keypoints(new_image)

        # TODO find correspondance between keypoints in adjacent frames
        correspondance = find_correspondance(old_keypoints, new_keypoints)

        # TODO find transform between adjacent frames
        transform = convert_correspondance_to_transform(correspondance)

        # TODO map transforms to camera displacement and rotation
        pose_delta = convert_transform_to_odometry(transform)

        # Save data that will be used for computing next delta
        self.last_image = new_image
        self.last_keypoints = new_keypoints

        return pose_delta


def main():

    # TODO read in video or list of images
    images = []

    pose_deltas = list(PoseEstimator(images))

    # TODO convert the list of deltas into displacement from start
    trajectory = accumulate(pose_deltas)

    # TODO save trajectory and probably make plots

    raise NotImplementedError


if __name__ == "__main__":
    main()
