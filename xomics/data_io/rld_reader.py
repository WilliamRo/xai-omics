from roma import console
from xomics.data_io.npy_reader import NpyReader
from xomics.data_io.utils.raw_rw import rd_tags, rd_file, rd_file_itk, resize_image_itk
from xomics.data_io.utils.preprocess import calc_SUV

import os


class RLDReader(NpyReader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SUBJECT_NAME = 'sub'
    self.FILETYPE = 'nii.gz'
    self._resample_param = {}
    self.process_func = self.process
    self.loader = self.nii_loader

  def get_tags(self):
    tagpath = self._current_filepath.replace(self.FILETYPE, 'pkl')
    tags = rd_tags(tagpath)
    return tags

  def nii_loader(self, filepath, **kwargs):
    self._current_filepath = filepath
    if self._load_type == 'CT':
      self._data, self._resample_param = rd_file(filepath, nii_param=True)
    else:
      self._data = rd_file_itk(filepath)
      self._data = resize_image_itk(self._data, raw=False,
                                    size=self._resample_param['size'],
                                    origin=self._resample_param['origin'],
                                    spacing=self._resample_param['spacing'],
                                    direction=self._resample_param['direction'])
    return self.pre_process(**kwargs)

  def process(self, suv=False, **kwargs):
    if suv and self._load_type == 'PET':
      tags = self.get_tags()
      self._data = calc_SUV(self._data, tags=tags, advance=True)


