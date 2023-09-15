import os

from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.raw_rw import rd_tags

class RLDReader(NpyReader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SUBJECT_NAME = 'sub'

  def _pre_process(self):
    tagpath = self._current_filepath[:-3] + 'pkl'
    tags = rd_tags(tagpath)
    return
