import os

from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.raw_rw import rd_tags
from xomics.data_io.utils.preprocess import calc_SUV

class RLDReader(NpyReader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SUBJECT_NAME = 'sub'

  def pre_process(self, suv=False, **kwargs):
    if suv:
      tagpath = self._current_filepath[:-3] + 'pkl'
      tags = rd_tags(tagpath)
      self._data = calc_SUV(self._data, tags=tags, advance=True)


    return
