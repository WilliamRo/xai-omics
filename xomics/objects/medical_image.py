from collections import OrderedDict
from roma import Nomear, console

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

    self._init_data()
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
    with open(filepath, 'wb') as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


  def save_as_nii(self, dir_path):
    dir_path = os.path.join(dir_path, self.key)
    if not os.path.exists(dir_path): os.mkdir(dir_path)

    # for key in self.images.keys():
    #   nifti_image = nib.Nifti1Image(self.images[key], affine=np.eye(4))
    #   file_path = os.path.join(dir_path, key + '.nii')
    #   nib.save(nifti_image, file_path)
    #   console.show_status(f'Saveing {file_path}')
    #
    # for key in self.labels.keys():
    #   nifti_label = nib.Nifti1Image(self.labels[key], affine=np.eye(4))
    #   file_path = os.path.join(dir_path, key + '.nii')
    #   nib.save(nifti_label, file_path)
    #   console.show_status(f'Saveing {file_path}')

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
      assert image.dtype == np.float32

    for label in self.labels.values():
      assert self.representative.shape == label.shape
      assert label.dtype == np.int8


  def window(self, layer: str, bottom, top):
    assert layer in self.images.keys()
    self.images[layer][self.images[layer] < bottom] = bottom
    self.images[layer][self.images[layer] > top] = top


  def normalization(self, layers):
    for layer in layers:
      assert layer in self.images.keys()
      mean = np.mean(self.images[layer])
      std = np.std(self.images[layer])
      self.images[layer] = ((self.images[layer] - mean) / std)


  def crop(self, crop_size: list, random_crop: bool):
    '''

    '''
    assert len(crop_size) == 3

    # Find the coordinates of the region with value 1
    indices = np.argwhere(self.labels['label-0'] == 1)
    max_indices = np.max(indices, axis=0)
    min_indices = np.min(indices, axis=0)

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


