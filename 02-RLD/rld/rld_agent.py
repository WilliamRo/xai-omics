from roma import console
from tframe.data.base_classes import DataAgent
from rld.rld_set import RLDSet
from xomics.objects.jutils.objects import GeneralMI

import os
import numpy as np



class RLDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    ds: RLDSet = cls.load_as_tframe_data(data_dir)
    test_pid = ['YHP00012417', 'YHP00010651', 'YHP00012231',
                'YHP00012016', 'YHP00011840', 'YHP00011890',
                'YHP00011239', 'YHP00011561', 'YHP00011818',
                'YHP00011905']
    train, test = ds.subset(test_pid, 'Test-Set')
    train, valid = train.split(-1, validate_size,
                               names=['Train-Set', 'Val-Set'])
    # train.mi_data.get_stat()
    # valid.mi_data.get_stat()
    # test.mi_data.get_stat()
    return train, valid, test

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    from rld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    data_path = os.path.join(data_dir, th.data_kwargs['dataset'], 'rld_data.csv')

    img_keys = th.data_set # + ['CT_seg']
    if not th.noCT:
      img_keys += ['CT']

    img_keys += th.extra_data

    img_type = {
      'CT': ['CT'],
      'PET': ['30G', '20S', '40S',
              '60G-1', '60G-2', '60G-3',
              '90G', '120S', '120G', '240G', '240S'],
      'MASK': ['CT_seg'],
      'STD': th.data_set[:1],
    }

    mi = GeneralMI.init_data(img_keys, data_path, img_type, th.process_param)
    return RLDSet(mi_data=mi, buffer_size=th.buffer_size)

