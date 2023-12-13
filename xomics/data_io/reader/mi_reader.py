from roma import console
from xomics import MedicalImage
from xomics.data_io.utils.preprocess import norm_size, normalize, crop_by_margin


import os
import numpy as np




class MiReader:

  def __init__(self, datadir: str = None):
    self.SUBJECT_NAME = 'subject'
    self.FILETYPE = 'npy'
    self._data = None
    self._load_type = None

    self.data = None
    self.datadir = datadir

    self.process_func = None
    self.loader = self.npy_load

    self.norm_list = None
    self.file_path_list = []
    self.dict = {}
    self.times = None
    self.now = 0

    self.methods_dict = {
      'sub': self.load_data_by_subject,
      'type': self.load_data_by_types,
      'train': self.load_data_with_same_norm,
    }

  def _load_data(self, subjects, types, methods, **kwargs):
    assert methods in self.methods_dict.keys()

    return self.methods_dict[methods](subjects, types, **kwargs)

  def _loader(self, filepath, *args, **kwargs):
    self.now += 1
    self.file_path_list.append(filepath)
    return self.loader(filepath, *args, **kwargs)

  def get_file_path(self, sub: int, types: str):
    return os.path.join(self.datadir, f'{self.SUBJECT_NAME}{sub}',
                        f'{self.SUBJECT_NAME}{sub}_{types}.'
                        f'{self.FILETYPE}')

  def load_data_by_subject(self, subjects,
                           types_list, **kwargs):
    mis = []
    type_strs = ["_".join(i) for i in types_list]
    for subject in subjects:
      data = {}
      console.supplement(f'loading sub{subject} with {type_strs}', level=2)
      for i, types_str in enumerate(type_strs):
        console.print_progress(i, len(type_strs))
        self._load_type = types_list[i][0]
        filepath = self.get_file_path(subject, types_str)
        self._loader(filepath, **kwargs)
        data[types_str] = self._data
      mi = MedicalImage(f'sub-{subject}', data)
      mis.append(mi)
    self.data = mis
    return mis

  def load_data_by_types(self, subjects,
                         types_list, **kwargs):
    mis = []

    for types in types_list:
      data = {}
      types_str = '_'.join(types)
      console.supplement(f'loading {types_str} with sub{subjects}', level=2)
      self._load_type = types[0]
      for i, subject in enumerate(subjects):
        console.print_progress(i, len(subjects))
        filepath = self.get_file_path(subject, types_str)
        self._loader(filepath, **kwargs)
        data[f'sub-{subject}'] = self._data
      mi = MedicalImage(f'{types_str}', data)
      mis.append(mi)
    self.data = mis
    return mis

  def load_data_with_same_norm(self, subjects,
                               types_list,
                               norm_types, **kwargs):
    mis = []
    self.norm_list = [None] * len(subjects)
    for types in types_list:
      self._load_type = types[0]
      type_str = '_'.join(types)

      if self._load_type not in norm_types:
        mis += self.load_data_by_types(subjects, [types], **kwargs)
        continue

      data_dict = {}
      console.supplement(f'loading {type_str} with sub{subjects}', level=2)
      for i, subject in enumerate(subjects):
        console.print_progress(i, len(subjects))
        filepath = self.get_file_path(subject, type_str)
        if self.norm_list[i] is None:
          data, self.norm_list[i] = self._loader(filepath, ret_norm=True, **kwargs)
        else:
          data = self._loader(filepath, norm=self.norm_list[i], **kwargs)
        data_dict[f'sub-{subject}'] = data
      mi = MedicalImage(type_str, data_dict)
      mis.append(mi)
    self.data = mis
    return mis

  def load_data(self, subjects,
                types,
                methods: str,
                **kwargs):
    """
    :param types: type in order
    :param subjects:
    :param methods: load methods
    :return: data
    """
    self.dict = {
      'sub': subjects,
      'type': types,
      'method': methods
    }
    self.times = len(subjects) * len(types)
    return self._load_data(subjects, types, methods, **kwargs)

  def npy_load(self, filepath, show_log=False, **kwargs):
    self._data = np.load(filepath)
    if show_log:
      console.supplement(f'Loaded `{os.path.split(filepath)[-1]}`', level=2)
    return self.pre_process(**kwargs)

  def pre_process(self,
                  norm=None, shape=None,
                  raw=False, clip=None,
                  ret_norm=False, crop_margin=None,
                  norm_method=None,
                  **kwargs):
    self.process_func(**kwargs)
    if crop_margin is not None:
      self._data = crop_by_margin(self._data, crop_margin)
    if not raw:
      if ret_norm:
        self._data, norm = normalize(self._data, norm_method, norm, ret_norm=ret_norm)
      else:
        self._data = normalize(self._data, norm_method, norm)
    if shape is not None:
      self._data = norm_size(self._data, shape)
    if clip is not None:
      self._data = np.clip(self._data, clip[0], clip[1])
    self._data = np.expand_dims(self._data, axis=-1)
    if ret_norm:
      return self._data, norm
    else:
      return self._data



if __name__ == '__main__':
  filePath = '../../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  reader = MiReader(filePath)
  img = reader.load_data([2], [['Full'], ['1-2']],
                         methods='sub', shape=[672, 440, 440])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
