from .correspondances import Correspondance


class ImageTransform:
    pass


def convert_correspondance_to_transform(correspondance: Correspondance) -> ImageTransform:
    # TODO
    raise NotImplementedError


class PoseDelta:
    pass


def convert_transform_to_odometry(transform: ImageTransform) -> PoseDelta:
    # TODO
    raise NotImplementedError
