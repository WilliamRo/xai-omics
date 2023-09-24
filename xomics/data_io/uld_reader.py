import numpy as np

from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.preprocess import calc_SUV, get_color_data
from xomics.data_io.utils.raw_rw import rd_file

import os



class UldReader(NpyReader):

  def __init__(self, datadir: str = None):
    super().__init__(datadir)
    self.size_list = []
    self.param_list = []
    self.process_func = self.process

  def process(self, use_suv=False, cmap=None):
    if use_suv:
      tags = self.load_tags()
      self._data = calc_SUV(self._data, tags)
    if cmap:
      self._data = get_color_data(self._data, cmap)

  def load_tags(self):
    tmp = os.path.dirname(self._current_filepath)
    file = os.path.basename(self._current_filepath)
    tagpath = os.path.join(tmp, f'tags_{file[:-3]}txt')
    tags = {}
    with open(tagpath, 'r') as f:
      for data in f.readlines():
        data = data.split(',')
        tags[data[0]] = float(data[1])
    return tags

  # todo: rubbish code to be improved (maybe)
  @classmethod
  def load_as_npy_data(cls, dirpath, file_list: list,
                       name_mask: (str, str), **kwargs):
    reader = cls()
    arr = []
    p_arr = []
    for file in file_list:
      filename = name_mask[0] + str(file) + name_mask[1]
      filepath = os.path.join(dirpath, filename)
      data, param = rd_file(filepath, nii_param=True)
      reader.size_list.append(data.shape)
      reader._data = data
      arr.append(reader.pre_process(**kwargs))
      p_arr.append(param)
    reader.data = arr
    reader.param_list = p_arr
    return reader




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
