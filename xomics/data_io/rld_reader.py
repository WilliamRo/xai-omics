import os

from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.raw_rw import rd_tags
from xomics.data_io.utils.preprocess import calc_SUV

class RLDReader(NpyReader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SUBJECT_NAME = 'sub'
    self.process_func = self.process

  def get_tags(self):
    tagpath = self._current_filepath[:-3] + 'pkl'
    tags = rd_tags(tagpath)
    return tags

  def process(self, suv=False, **kwargs):
    if suv and self._load_type == 'PET':
      tags = self.get_tags()
      self._data = calc_SUV(self._data, tags=tags, advance=True)


