from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def cut_mix(images: torch.Tensor, pointmap: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies cutmix to a batch of images and pointmap.

  Args:
    images: torch.Tensor of shape [B,C,H,W]
    pointmap: torch.Tensor of shape [B,H,W]
    alpha: float, the beta distribution parameter

  Returns:
    Tuple of torch.Tensor: The modified images and pointmap.
  """
  lam = np.random.beta(alpha, alpha)
  index = torch.randperm(images.shape[0])

  cut_rat = np.sqrt(1. - lam)
  cut_w = int(images.size(2) * cut_rat)
  cut_h = int(images.size(3) * cut_rat)

  # Uniformly sample the centre of the patch
  cx = np.random.randint(images.size(2))
  cy = np.random.randint(images.size(3))

  # Calculate the patch coordinates
  bbx1 = np.clip(cx - cut_w // 2, 0, images.size(2))
  bby1 = np.clip(cy - cut_h // 2, 0, images.size(3))
  bbx2 = np.clip(cx + cut_w // 2, 0, images.size(2))
  bby2 = np.clip(cy + cut_h // 2, 0, images.size(3))

  # Apply CutMix
  images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
  pointmap[:, bbx1:bbx2, bby1:bby2] = pointmap[index, bbx1:bbx2, bby1:bby2]
  return images, pointmap

