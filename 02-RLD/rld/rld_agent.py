from tframe.data.base_classes import DataAgent
from rld.rld_set import RLDSet

import os



class RLDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    from rld_core import th
    ds: RLDSet = cls.load_as_tframe_data(data_dir)
    return ds.split(-1, validate_size, test_size,
                    names=['Train-Set', 'Val-Set', 'Test-Set'])

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    from rld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    data_root = os.path.join(data_dir, th.data_kwargs['dataset'])
    return RLDSet.load_as_rldset(data_root)

