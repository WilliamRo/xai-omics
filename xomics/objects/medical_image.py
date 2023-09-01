from collections import OrderedDict
from roma import Nomear
from tframe import local
from tframe import console

import numpy as np
import pickle



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
  def size(self):
    return self.representative.shape

  @property
  def num_layers(self): return len(self.images)

  # endregion: Properties

  # region: Public Mehtods

  def save(self, filepath):
    if filepath.split('.')[-1] != self.EXTENSION:
      filepath += '.{}'.format(self.EXTENSION)
    with open(filepath, 'wb') as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


  @classmethod
  def load(self, path):
    assert isinstance(path, str)

    with open(path, 'rb') as input:
      # console.show_status('Loading `{}` ...'.format(path))
      return pickle.load(input)


  def get_bottom_top(self, center: list, crop_size: list):
    bottom_z = center[0] - crop_size[0] // 2
    bottom_x = center[1] - crop_size[1] // 2
    bottom_y = center[2] - crop_size[2] // 2

    bottom_z = 0 if bottom_z < 0 else bottom_z
    bottom_x = 0 if bottom_x < 0 else bottom_x
    bottom_y = 0 if bottom_y < 0 else bottom_y

    top_z = bottom_z + crop_size[0]
    top_x = bottom_x + crop_size[1]
    top_y = bottom_y + crop_size[2]

    top_z = self.size[0] if top_z > self.size[0] else top_z
    top_x = self.size[1] if top_x > self.size[1] else top_x
    top_y = self.size[2] if top_y > self.size[2] else top_y

    bottom_z = top_z - crop_size[0]
    bottom_x = top_x - crop_size[1]
    bottom_y = top_y - crop_size[2]

    return [bottom_z, bottom_x, bottom_y], [top_z, top_x, top_y]


  # endregion: Public Mehtods

  # region: Private Mehtods

  def _check_data(self):
    for image in self.images.values():
      assert self.representative.shape == image.shape

    for label in self.labels.values():
      assert self.representative.shape == label.shape


  def window(self, layer: str, bottom, top):
    assert layer in self.images.keys()
    self.images[layer][self.images[layer] < bottom] = bottom
    self.images[layer][self.images[layer] > top] = top


  def normalization(self, layers):
    for layer in layers:
      assert layer in self.images.keys()
      mean = np.mean(self.images[layer])
      std = np.mean(self.images[layer])
      self.images[layer] = (self.images[layer] - mean) / std


  def crop(self, crop_size: list):
    '''

    '''
    assert len(crop_size) == 3

    # Find the coordinates of the region with value 1
    indice = np.argwhere(self.labels['label-0'] == 1)
    max, min = np.max(indice, axis=0), np.min(indice, axis=0)

    # Calculate the center coordinates of the region
    center = [(max[i] + min[i]) // 2 for i in range(len(max))]

    # Calculate the bottom and the top
    bottom, top = self.get_bottom_top(center, crop_size)

    # Crop
    for key in self.images.keys():
      self.images[key] = self.images[key][
                         bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]]

    for key in self.labels.keys():
      self.labels[key] = self.labels[key][
                         bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]]


# endregion: Private Mehtods


