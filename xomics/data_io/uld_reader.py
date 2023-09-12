from typing import Union
from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.preprocess import calc_SUV, get_color_data

import os
import numpy as np



class UldReader(NpyReader):

  def __init__(self, datadir: str = None):
    super().__init__(datadir)

  def _pre_process(self, use_suv=False, cmap=None):
    if use_suv:
      tags = self.load_tags()
      self._data = calc_SUV(self._data, tags)
    if cmap:
      self._data = get_color_data(self._data, cmap)

  def load_tags(self):
    tmp = self._current_filepath
    tagpath = os.path.join(tmp[0], f'tags_{tmp[1][:-3]}txt')
    tags = {}
    with open(tagpath, 'r') as f:
      for data in f.readlines():
        data = data.split(',')
        tags[data[0]] = float(data[1])
    return tags






if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  reader = UldReader(filePath)
  img = reader.load_data([2, 3], [['Full'], ['1-2']], methods='type',
                         shape=[1, 608, 440, 440, 1])

  print(img)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
