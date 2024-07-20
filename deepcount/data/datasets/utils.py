import os 

import cv2 
import torch 
import numpy as np

def read_image(path: str, directory: str = None) -> np.ndarray:
    """Reads an image from a specified path or directory and returns it as a numpy array.

    This function attempts to read the image from the given directory. If it fails,
    it tries to read from the specified path directly.

    Args:
        path: A string specifying the path to the image.
        directory: An optional string specifying the base directory of the image. If None,
            the function will attempt to read the image directly from the path.

    Returns:
        A numpy array representing the read image, converted to RGB format.
    """

    full_path = os.path.join(directory, path) if directory else path
    image = cv2.imread(full_path) if cv2.imread(full_path) else cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {full_path} or {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def apply_transforms(image: np.ndarray, points: np.ndarray, transforms:callable, tolerance: int):
    """Applies a set of transformations to an image and its corresponding points.

    This function modifies the input image and points based on the provided transformations.
    It also adjusts the points based on a specified tolerance level.

    Args:
        image: A numpy array representing the image to be transformed of shape [H,W] or [C,H,W].
        points: A numpy array of points to be transformed alongside the image of shape [points, 2].
        transforms: A callable or series of callables that apply transformations to the image and points.
        tolerance: An integer specifying the tolerance level for point transformations.

    Returns:
        The transformed image and points as a tuple (transformed_image, transformed_points).
    """
    data = transforms(image=image, keypoints=points)
    image = data['image']
    points = torch.floor(torch.tensor(data['keypoints'])).to(int)
    kpoints = torch.zeros_like(image)[0, :, :]
    if len(points) > 0:
        points_x = points[:, 0]
        points_y = points[:, 1]
        if (points_x.min() < -1 or points_y.min() < -1) or (points_x.max() > kpoints.shape[0] + tolerance or points_y.max() > kpoints.shape[1] + tolerance):
            # This was added to fix some issues where the points are slightly out of range (tolerance pixel) [e.g. case image size was 224x224 and the point was 218.196x23.999]
            assert False, f"points out of range {points_x.min()} {points_y.min()} {points_x.max()} {points_y.max()}"
        points_x.clamp_(0, kpoints.shape[0] - 1)
        points_y.clamp_(0, kpoints.shape[1] - 1)
        kpoints[points_x, points_y] = 1

    return image, kpoints
