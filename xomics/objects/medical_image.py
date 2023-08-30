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


  # endregion: Public Mehtods

  # region: Private Mehtods

  def _check_data(self):
    for image in self.images.values():
      assert self.representative.shape == image.shape

    for label in self.labels.values():
      assert self.representative.shape == label.shape

  # endregion: Private Mehtods


