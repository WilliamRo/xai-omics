from xomics.data_io.reader.mi_reader import MiReader
from xomics.data_io.utils.raw_rw import rd_tags, rd_file, rd_file_itk, \
  resize_image_itk, itk_norm
from xomics.data_io.utils.preprocess import calc_SUV, reverse_suv


class RLDReader(MiReader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SUBJECT_NAME = 'sub'
    self.FILETYPE = 'nii.gz'

    self.img_param = []
    self.process_func = self.process
    self.loader = self.nii_loader

  def get_tags(self, index):
    tagpath = self.file_path_list[index].replace(self.FILETYPE, 'pkl')
    tags = rd_tags(tagpath)
    return tags

  def get_raw_data(self, data, index, rev_suv=False):
    tags = self.get_tags(index)
    norm = self.norm_list[index]
    if rev_suv:
      return reverse_suv(data * (norm[0] - norm[1]) + norm[1], tags)
    else:
      return data * (norm[0] - norm[1]) + norm[1]

  def nii_loader(self, filepath, noCT=False, **kwargs):
    types = len(self.dict['type']) - noCT
    subs = len(self.dict['sub']) - noCT
    if self._load_type == 'CT':
      self._data = rd_file_itk(filepath)
      i = (self.now - 1) % (types - 1)
      self._data = itk_norm(self._data, 255)
      self._data = resize_image_itk(self._data, raw=False,
                                    size=self.img_param[i]['size'],
                                    origin=self.img_param[i]['origin'],
                                    spacing=self.img_param[i]['spacing'],
                                    direction=self.img_param[i]['direction'])
    else:
      self._data, img_param = rd_file(filepath, nii_param=True)
      # assume: all pet image's param of a sub is same
      if self.dict['method'] in ['type', 'train']:
        if self.now % subs == 1:
          self.img_param.append(img_param)
      elif self.dict['method'] in ['sub']:
        if self.now % types == 1:
          self.img_param.append(img_param)

    return self.pre_process(**kwargs)

  def process(self, suv=False, **kwargs):
    self._data = self._data
    if suv and self._load_type == 'PET':
      tags = self.get_tags(-1)
      self._data = calc_SUV(self._data, tags=tags, advance=True)

  def export_nii(self, data, filepath, **kwargs):
    from xomics.data_io.utils.raw_rw import wr_file
    return wr_file(data, filepath, **kwargs)

  def load_nii(self, filepath):
    from xomics.data_io.utils.raw_rw import rd_file
    return rd_file(filepath)

  def load_name(self, sub):
    pass

