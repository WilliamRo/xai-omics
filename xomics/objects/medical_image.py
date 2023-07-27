from collections import OrderedDict
from roma import Nomear

import numpy as np



class MedicalImage(Nomear):
  """The first dimension is always slice dimension"""

  def __init__(self, key='noname', images=None, labels=None):
    self.key = key

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

  def save(self, path):
    pass

  # endregion: Public Mehtods

  # region: Private Mehtods

  def _check_data(self):
    pass

  # endregion: Private Mehtods


