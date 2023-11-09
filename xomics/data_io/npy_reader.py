from roma import console
from xomics import MedicalImage
from xomics.data_io.utils.preprocess import norm_size, normalize


import os
import numpy as np




class NpyReader:

  def __init__(self, datadir: str = None):
    self.SUBJECT_NAME = 'subject'
    self.FILETYPE = 'npy'
    self._data = None
    self._current_filepath = None
    self._load_type = None
    self.data = None
    self.datadir = datadir
    self.process_func = None
    self.loader = self.npy_load
    self.methods_dict = {
      'sub': self.load_data_by_subject,
      'type': self.load_data_by_types,
      'mi': self.load_mi_data,
      'pair': self.load_data_in_pair,
      'train': self.load_data_with_same_norm,
    }

  def _load_data(self, subjects, types, methods, **kwargs):
    assert methods in self.methods_dict.keys()

    return self.methods_dict[methods](subjects, types, **kwargs)

  def get_file_path(self, sub: int, types: str):
    return os.path.join(self.datadir, f'{self.SUBJECT_NAME}{sub}',
                        f'{self.SUBJECT_NAME}{sub}_{types}.'
                        f'{self.FILETYPE}')

  def load_data_by_subject(self, subjects: list[int],
                           types_list: list[list[str]], **kwargs):
    arr_dict = {}
    type_strs = ["_".join(i) for i in types_list]
    for subject in subjects:
      arr = []
      console.supplement(f'loading sub{subject} with {type_strs}', level=2)
      for i, types_str in enumerate(type_strs):
        console.print_progress(i, len(type_strs))
        self._load_type = types_list[i][0]
        filepath = self.get_file_path(subject, types_str)
        self.loader(filepath, **kwargs)
        arr.append(self._data)
      arr_dict[subject] = arr
    self.data = arr_dict
    return arr_dict

  def load_data_by_types(self, subjects: list[int],
                         types_list: list[list[str]], **kwargs):
    arr_dict = {}

    for types in types_list:
      arr = []
      types_str = '_'.join(types)
      console.supplement(f'loading {types_str} with sub{subjects}', level=2)
      self._load_type = types[0]
      for i, subject in enumerate(subjects):
        console.print_progress(i, len(subjects))
        filepath = self.get_file_path(subject, types_str)
        self.loader(filepath, **kwargs)
        arr.append(self._data)
      arr_dict[types_str] = arr
    self.data = arr_dict
    return arr_dict

  def load_data_in_pair(self, subjects: list[int],
                        types_list: list[list[str]], **kwargs):
    assert len(types_list) == 2
    features = []
    targets = []
    for subject in subjects:
      type_str1 = '_'.join(types_list[0])
      type_str2 = '_'.join(types_list[1])
      assert types_list[0][0] == types_list[1][0]
      self._load_type = types_list[0][0]
      f_filepath = self.get_file_path(subject, type_str1)
      t_filepath = self.get_file_path(subject, type_str2)

      feature, norm = self.loader(f_filepath, ret_norm=True, **kwargs)
      self.loader(t_filepath, norm=norm, **kwargs)
      target = self._data

      features.append(feature)
      targets.append(target)

    self.data = {
      'features': features,
      'targets': targets,
    }
    return self.data

  def load_data_with_same_norm(self, subjects: list[int],
                               types_list: list[list[str]],
                               norm_types: list[str], **kwargs):
    data_dict = {}
    for types in types_list:
      norms = [None] * len(subjects)
      self._load_type = types[0]
      type_str = '_'.join(types)

      if self._load_type not in norm_types:
        data_dict[type_str] = self.load_data_by_types(subjects, [types],
                                                      **kwargs)[type_str]
        continue

      data_list = []
      console.supplement(f'loading {type_str} with sub{subjects}', level=2)
      for i, subject in enumerate(subjects):
        console.print_progress(i, len(subjects))
        filepath = self.get_file_path(subject, type_str)
        if norms[i] is None:
          data, norm = self.loader(filepath, ret_norm=True, **kwargs)
        else:
          data = self.loader(filepath, norm=norms[i], **kwargs)
        data_list.append(data)
      data_dict[type_str] = data_list

    self.data = data_dict
    return data_dict

  def load_data(self, subjects: list[int],
                types: list[list[str]],
                methods: str,
                **kwargs) -> dict:
    """
    :param types: type in order
    :param subjects:
    :param methods: load methods
    :return: data
    """
    return self._load_data(subjects, types, methods, **kwargs)

  def load_mi_data(self, subjects: list[int],
                   types_list: list[list[str]], **kwargs):
    mis = []
    imgs = self.load_data(subjects, types_list, 'sub', mi=True, **kwargs)
    type_strs = ['_'.join(types) for types in types_list]
    for subject in subjects:
      data = dict(zip(type_strs, imgs[subject]))
      mi = MedicalImage(f'{self.SUBJECT_NAME}-{subject}', data)
      mis.append(mi)
    return mis

  def npy_load(self, filepath, show_log=False, **kwargs):
    self._current_filepath = filepath
    self._data = np.load(filepath)
    if show_log:
      console.supplement(f'Loaded `{os.path.split(filepath)[-1]}`', level=2)
    return self.pre_process(**kwargs)

  def pre_process(self,
                  norm=None, shape=None,
                  raw=False, clip=None,
                  ret_norm=False, mi=False,
                  norm_margin=None, swapaxe=None,
                  **kwargs):
    self.process_func(**kwargs)
    if not raw:
      if ret_norm:
        self._data, norm = normalize(self._data, norm,
                                     margin=norm_margin, ret_norm=ret_norm)
      else:
        self._data = normalize(self._data, norm, margin=norm_margin)
    if shape is not None:
      self._data = norm_size(self._data, shape)
    if clip is not None:
      self._data = np.clip(self._data, clip[0], clip[1])
    if mi:
      self._data = self._data.reshape(self._data.shape[1:])
    if swapaxe is not None:
      for axis in swapaxe:
        self._data = self._data.swapaxes(axis[0], axis[1])
    if ret_norm:
      return self._data, norm
    else:
      return self._data



if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  reader = NpyReader(filePath)
  img = reader.load_data([2], [['Full'], ['1-2']],
                         methods='sub', shape=[1, 672, 440, 440, 1])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
