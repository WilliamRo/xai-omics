from roma import console
from tframe.data.base_classes import DataAgent
from rld.rld_set import RLDSet
from xomics.objects.jutils.objects import GeneralMI

import os
import numpy as np



class RLDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    from rld_core import th
    ds: RLDSet = cls.load_as_tframe_data(data_dir)
    with open(os.path.join(data_dir,
                           th.data_kwargs['dataset'] ,'test_id'), 'r') as f:
      test_pid = f.read().split('\n')
      if test_pid[-1] == '':
        test_pid = test_pid[:-1]
    test_pid = [f'YHP000{pid}' for pid in test_pid]
    train, test = ds.subset(test_pid, 'Test-Set')
    test.buffer_size = len(test)

    if th.gan:
      valid_pid = [
        'YHP00012350', 'YHP00010521', 'YHP00011030', 'YHP00012439',
        'YHP00012490'
      ]
      train, valid = train.subset(valid_pid, 'Val-Set')
      train.name = 'Train-Set'
    else:
      train, valid = train.split(-1, validate_size,
                                 names=['Train-Set', 'Val-Set'])

    print("Validation Pids: ", valid.mi_data.pid)
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
    if th.gen_mask:
      img_keys += ['CT_seg']

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

