import numpy as np

from tframe.data.base_classes import DataAgent
from uld.uld_set import ULDSet

import os



class ULDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    ds: ULDSet = cls.load_as_tframe_data(data_dir)

    return ds.split(-1, validate_size, test_size,
                    names=['Train-Set', 'Val-Set', 'Test-Set'])


  @classmethod
  def load_as_tframe_data(cls, data_dir) -> ULDSet:
    # features/targets.shape = [N, S, H, W, 1]
    features, targets = cls.load_as_numpy_arrays(data_dir)
    return ULDSet(features, targets)


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    from uld_core import th

    data_root = os.path.join(data_dir, th.data_config)

    features = np.zeros(shape=[6, 644, 512, 512, 1])

    return features, features