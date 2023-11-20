from tqdm import tqdm
from xomics.objects import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from typing import Union
from functools import reduce

import numpy as np
import random
import math
import os



def add_gaussian_noise(image: np.ndarray, mean=0, std=1):
  assert len(image.shape) in [2, 3]

  noise = np.random.normal(mean, std, image.shape)
  return image + noise


def image_rotation(image, angle):
  assert len(image.shape) in [2, 3]
  assert angle in [0, 90, 180, 270]
  for _ in range(3):
    if angle <= 0: break
    image = [np.rot90(img) for img in image]
    angle = angle - 90

  return np.array(image)


def image_flip(image, axes):
  assert len(image.shape) in [2, 3]
  return np.flip(image, axis=axes)


def crop_3d(input_data: list, crop_size: list,
            random_crop: bool, basis: list):

  assert len(crop_size) == 3

  assert all(isinstance(arr, np.ndarray) for arr in input_data)
  assert all(arr.shape == input_data[0].shape for arr in input_data)
  raw_shape = input_data[0].shape

  # Find the coordinates of the region with value 1
  union = reduce(np.logical_or, basis)
  indices = np.argwhere(union == 1)
  max_indices = np.max(indices, axis=0)
  min_indices = np.min(indices, axis=0)

  # Make sure that the crop size > delta indice
  delta_indices = [
    maxi - mini for maxi, mini in zip(max_indices, min_indices)]
  assert all(d <= c for d, c in zip(delta_indices, crop_size))

  # Calculate the bottom and the top
  bottom, top = get_bottom_top(
    max_indices, min_indices, raw_shape, crop_size, random_crop)

  # Crop
  output_data = []
  for arr in input_data:
    output_data.append(
      arr[bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]])

  return output_data


def get_bottom_top(max_indices, min_indices,
                   raw_shape, new_shape, random_crop):
  center = [
    (maxi + mini) // 2 for maxi, mini in zip(max_indices, min_indices)]

  if random_crop:
    bottom = [random.randint(max(0, maxi - n), min(mini, r - n))
              for maxi, mini, r, n in
              zip(max_indices, min_indices, raw_shape, new_shape)]
    top = [b + n for b, n in zip(bottom, new_shape)]
  else:
    bottom = [max(0, c - n // 2) for c, n in zip(center, new_shape)]
    top = [min(b + n, r) for b, n, r in zip(bottom, new_shape, raw_shape)]
    bottom = [t - n for t, n in zip(top, new_shape)]

  return bottom, top



if __name__ == "__main__":
  pass

