from collections import OrderedDict
from roma import Nomear, console
from typing import Union

import os
import nibabel as nib
import SimpleITK as sltk
import numpy as np
import pickle
import random



class MedicalImage(Nomear):
  """The first dimension is always slice dimension"""

  def __init__(self, key='noname', images=None, labels=None):
    self.key = key
    self.EXTENSION = 'mi'

    if images is None: images = OrderedDict()
    if labels is None: labels = OrderedDict()

    self.images = images
    self.labels = labels

    self._check_data()

  # region: Properties

  @property
  def representative(self) -> np.ndarray:
    return list(self.images.values())[0]

  @property
  def num_slices(self):
    return self.representative.shape[0]

  @property
  def shape(self):
    return self.representative.shape

  @property
  def num_layers(self): return len(self.images)

  # endregion: Properties

  # region: Public Methods

  def save(self, filepath):
    if filepath.split('.')[-1] != self.EXTENSION:
      filepath += '.{}'.format(self.EXTENSION)

    # image
    for k in self.images.keys():
      self.images[k] = self.images[k].astype(np.float16)

    # label
    for k in self.labels.keys():
      self.labels[k] = self.labels[k].astype(np.int8)

    with open(filepath, 'wb') as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


  def save_as_nii(self, dir_path):
    dir_path = os.path.join(dir_path, self.key)
    if not os.path.exists(dir_path): os.mkdir(dir_path)

    for key in self.images.keys():
      file_path = os.path.join(dir_path, key + '.nii')
      sltk.WriteImage(sltk.GetImageFromArray(self.images[key]), file_path)
      console.show_status(f'Saveing {file_path}')

    for key in self.labels.keys():
      file_path = os.path.join(dir_path, key + '.nii')
      sltk.WriteImage(sltk.GetImageFromArray(self.labels[key]), file_path)
      console.show_status(f'Saveing {file_path}')


  @classmethod
  def load(cls, path):
    assert isinstance(path, str)

    with open(path, 'rb') as input:
      # console.show_status('Loading `{}` ...'.format(path))
      loaded_data = pickle.load(input)

    loaded_data._init_data()
    loaded_data._check_data()

    return loaded_data


  def get_bottom_top(self, max_indices, min_indices,
                     raw_shape, new_shape, random_crop):

    center = [(maxi + mini) // 2
              for maxi, mini in zip(max_indices, min_indices)]

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


  # endregion: Public Methods

  # region: Private Methods

  def _init_data(self):
    for key in self.images.keys():
      self.images[key] = np.float32(self.images[key])

    for key in self.labels.keys():
      self.labels[key] = np.int8(self.labels[key])


  def _check_data(self):
    for image in self.images.values():
      assert self.representative.shape == image.shape

    for label in self.labels.values():
      assert self.representative.shape == label.shape


  def window(self, layer: str, bottom, top):
    assert layer in self.images.keys()
    self.images[layer][self.images[layer] < bottom] = bottom
    self.images[layer][self.images[layer] > top] = top


  def normalization(self, layers, method):
    def z_score(input):
      mean = np.mean(input)
      std = np.std(input)
      return (input - mean) / std


    def min_max(input):
      max_data = np.max(input)
      min_data= np.min(input)
      return (input - min_data) / (max_data - min_data)

    assert method in ['z_score', 'min_max']
    met = z_score if method == 'z_score' else min_max

    for layer in layers:
      assert layer in self.images.keys()
      self.images[layer] = met(self.images[layer])


  def crop(self, crop_size: list, random_crop: bool, basis: Union[str, list]):
    '''

    '''
    assert len(crop_size) == 3
    max_indices = [0, 0, 0]
    min_indices = list(self.shape)

    # Find the coordinates of the region with value 1
    if isinstance(basis, str):
      assert basis in self.labels.keys()
      indices = np.argwhere(self.labels[basis] == 1)
      max_indices = np.max(indices, axis=0)
      min_indices = np.min(indices, axis=0)
    elif isinstance(basis, list):
      for b in basis:
        assert b in self.labels.keys()
        indices = np.argwhere(self.labels[b] == 1)
        max_indices = [
          max(a, b) for a, b in zip(np.max(indices, axis=0), max_indices)]
        min_indices = [
          min(a, b) for a, b in zip(np.min(indices, axis=0), min_indices)]



    delta_indices = [
      maxi - mini for maxi, mini in zip(max_indices, min_indices)]

    assert all(d <= c for d, c in zip(delta_indices, crop_size))

    # Calculate the bottom and the top
    bottom, top = self.get_bottom_top(
      max_indices, min_indices, self.shape, crop_size, random_crop)

    # Crop
    for key in self.images.keys():
      self.images[key] = self.images[key][
                         bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]]

    for key in self.labels.keys():
      self.labels[key] = self.labels[key][
                         bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]]


# endregion: Private Methods


