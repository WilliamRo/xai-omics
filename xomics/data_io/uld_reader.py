from typing import Union
from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.preprocess import calc_SUV, get_color_data

import os
import numpy as np



class UldReader(NpyReader):

  def __init__(self, datadir: str = None):
    super().__init__(datadir)
    self.tags = None
    self.conditions_dict[(list, dict)] = self.load_data_pair

  def _pre_process(self, use_suv=False, cmap=None):
    if use_suv:
      tags = self.load_tags()
      self._data = calc_SUV(self._data, tags)
    if cmap:
      self._data = get_color_data(self._data, cmap)
    if True in np.isnan(self._data):
      print("BAD DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!")

  def load_data(self, subjects: Union[int, str, list],
                doses: Union[str, list, dict],
                **kwargs):
    """
    support 4 ways to load data
    :param subjects:
    :param doses:
    :return: data
    """
    return self._load_data(subjects, doses, **kwargs)

  def load_tags(self):
    tmp = self._current_filepath
    tagpath = os.path.join(tmp[0], f'tags_{tmp[1][:-3]}txt')
    tags = {}
    with open(tagpath, 'r') as f:
      for data in f.readlines():
        data = data.split(',')
        tags[data[0]] = float(data[1])
    return tags

  def load_data_pair(self, subjects: list[int], doses: dict, **kwargs):
    features = []
    targets = []
    for subject in subjects:
      f_filepath = os.path.join(self.datadir, f'{self.SUBJECT_NAME}{subject}',
                                f'{self.SUBJECT_NAME}{subject}_{doses["feature"]}.npy')
      t_filepath = os.path.join(self.datadir, f'{self.SUBJECT_NAME}{subject}',
                                f'{self.SUBJECT_NAME}{subject}_{doses["target"]}.npy')

      self.npy_load(f_filepath, ret_norm=True, **kwargs)
      feature, norm = self._data
      self.npy_load(t_filepath, norm=norm, **kwargs)
      target = self._data

      features.append(feature)
      targets.append(target)

    self.raw_data = features, targets
    self.data = np.concatenate(features), np.concatenate(targets)
    return self.data




if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  reader = UldReader(filePath)
  img = reader.load_data(2, ['Full', '1-2'], shape=[1, 608, 440, 440, 1])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
