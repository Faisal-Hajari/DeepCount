from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, Targets
from .functional import * 


class ToTensor(BasicTransform):
    """Converts images/masks to PyTorch Tensors, inheriting from BasicTransform. Supports images in numpy `HWC` format
    and converts them to PyTorch `CHW` format. If the image is in `HW` format, it will be converted to PyTorch `HW`.
     
    with fixed error (negtive stride numpy arrays not supported)
    Attributes:
        transpose_mask (bool): If True, transposes 3D input mask dimensions from `[height, width, num_channels]` to
            `[num_channels, height, width]`.
        always_apply (bool): Deprecated. Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(self, transpose_mask: bool = False, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> dict[str, Any]:
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img: np.ndarray, **params: Any) -> torch.Tensor:
        if len(img.shape) not in [2, 3]:
            msg = "Albumentations only supports images in HW or HWC format"
            raise ValueError(msg)

        if len(img.shape) == MONO_CHANNEL_DIMENSIONS:
            img = np.expand_dims(img, 2)
            
        # we added .copy to different stride arrays error
        return torch.from_numpy(img.transpose(2, 0, 1).copy()) 
    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask.copy())

    def apply_to_masks(self, masks: list[np.ndarray], **params: Any) -> list[torch.Tensor]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("transpose_mask",)
    
class CutMix:
    """
    CutMix augmentation technique.
    inspired by CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    https://arxiv.org/abs/1905.04899

    Args:
        alpha (float): The hyperparameter alpha controls the strength of CutMix. 
            It determines the ratio of the second image to be mixed with the first image.

    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, data: torch.Tensor, pointmap: Tensor) -> Tuple[Tensor]:
        """
        Apply CutMix augmentation to the input data.

        Args:
            data (Tensor): The input data to be augmented.
            pointmap (Tensor): The keypoints associated with the input data.

        Returns:
            Tuple[Tensor]: The augmented data after applying CutMix.
        """
        return cut_mix(data, pointmap, self.alpha)

  