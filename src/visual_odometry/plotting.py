import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from visual_odometry.geometry import PoseDelta


def plot_poses(poses: list[PoseDelta], filename: Path, figsize) -> None:
    """Adapted from PA5 (Quinn Schebell).

    Poses are wTi (in world frame, which is defined as 0th camera frame)
    """
    axis_length = 0.5

    _, ax = plt.subplots(figsize=figsize)

    for i, delta in enumerate(poses):
        wTi = delta.se3_transform
        wti = wTi[:3, 3]

        # assume ground plane is xz plane in camera coordinate frame
        # 3d points in +x and +z axis directions, in homogeneous coordinates
        posx = wTi @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
        posz = wTi @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

        ax.plot([float(wti[0]), float(posx[0])], [float(wti[2]), float(posx[2])], "b", zorder=1)
        ax.plot([float(wti[0]), float(posz[0])], [float(wti[2]), float(posz[2])], "b", zorder=1)

        ax.scatter(float(wti[0]), float(wti[2]), 40, marker=".", color='g', zorder=2)

    plt.axis("equal")
    plt.title("Egovehicle trajectory")
    plt.xlabel("x camera coordinate (of camera frame 0)")
    plt.ylabel("z camera coordinate (of camera frame 0)")
    plt.savefig(str(filename))
