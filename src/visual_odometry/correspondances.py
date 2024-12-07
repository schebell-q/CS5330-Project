from dataclasses import dataclass

import numpy as np


@dataclass
class Correspondances:
    im1: np.array
    im2: np.array
    p1: np.array
    p2: np.array