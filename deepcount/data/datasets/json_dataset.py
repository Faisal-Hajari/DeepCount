import os 
import json
from typing import Tuple, Optional, Any

import numpy as np
from torch.utils.data import Dataset

from deepcount.data.datasets import utils

class JSONDataset:
    """A PyTorch-compatible dataset object for JSON-based datasets.

    This dataset reads images and points from a JSON file, applying optional
    transformations to the images and keypoints.

    Attributes:
        path (str): The path to the dataset directory.
        transforms (callable, optional): A function/transform that takes in an
            image and keypoints and returns a transformed version.
        json_file_name (str): The name of the JSON file containing annotations.
        data_key (str, optional): The key in the JSON file to access data. If None,
            the whole JSON content is used.
        image_key (str): The key for accessing image data in the JSON.
        point_key (str): The key for accessing point data in the JSON.
        tolerance (int): Tolerance parameter for transformations.

    Args:
        path (str): The path to the dataset directory.
        transforms (callable, optional): Optional transform to be applied
            on a sample.
        json_file_name (str): Name of the JSON file with annotations.
        data_key (str, optional): Specific key for accessing data in the JSON file.
        image_key (str): Key for accessing image data in the JSON.
        point_key (str): Key for accessing point data in the JSON.
        tolerance (int): Tolerance for transformations.
    """

    def __init__(self, path: str, transforms: Optional[callable] = None, json_file_name: str = "annotation.json",
                 data_key: str = None, image_key: str = "image", point_key: str = "points", tolerance: int = 1) -> None:
        super().__init__()
        self.path = path
        with open(os.path.join(path, json_file_name), "r") as file:
            self.data = json.load(file)[data_key] if data_key else json.load(file)
        self.image_key = image_key
        self.point_key = point_key
        self.transforms = transforms
        self.tolerance = tolerance

    def __getitem__(self, index: int) -> Tuple[Any, np.ndarray]:
        """Fetches the image and keypoints at the given index.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: (image, keypoints) where image is the image object and
            keypoints is a numpy array of key points.
        """
        image, points = self.data[index][self.image_key], self.data[index][self.point_key]
        kpoints = np.array(points)
        image = utils.read_image(os.path.join(self.path, image))
        if self.transforms:
            image, kpoints = utils.apply_transforms(image, kpoints, self.transforms, self.tolerance)
        return image, kpoints

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.data)