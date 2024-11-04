from itertools import accumulate

from .correspondances import find_correspondance
from .geometry import convert_correspondance_to_transform, convert_transform_to_odometry
from .keypoints import find_keypoints


def main():

    # TODO read in video or list of images
    images = []

    # TODO find keypoints in each frame
    keypoints = find_keypoints(images)

    # TODO find correpondances between keypoints in adjacent frames
    correspondances = [find_correspondance(*pts) for pts in zip(keypoints[:-1], keypoints[1:], strict=False)]

    # TODO find transform between adjacent frames
    transforms = map(convert_correspondance_to_transform, correspondances)

    # TODO map transforms to camera displacement and rotation
    pose_deltas = map(convert_transform_to_odometry, transforms)

    # TODO convert the list of deltas into displacement from start
    trajectory = accumulate(pose_deltas)

    # TODO save trajectory and probably make plots

    raise NotImplementedError


if __name__ == "__main__":
    main()
