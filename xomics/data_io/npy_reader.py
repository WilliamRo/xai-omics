from tframe import console
from typing import Union
from xomics.data_io.utils.preprocess import norm_size, normalize

import os
import numpy as np




class NpyReader:

  def __init__(self, datadir: str):
    self.SUBJECT_NAME = 'subject'
    self.datadir = datadir
    self.current_filepath = None
    self.data = None
    self._data = None
    self.conditions_dict = {
      (int, list): self.load_data_by_subject,
      (list, str): self.load_data_by_dose,
      (int, str): self.load_one_data,
    }

  def _load_data(self, subjects, doses, **kwargs):
    self.data = self.load_numpy_data(subjects, doses, **kwargs)
    return self.data

  def _npy_load(self, **kwargs):
    return self.pre_process(**kwargs)

  def _pre_process(self, **kwargs):
    pass

  def load_numpy_data(self, subjects, doses, **kwargs):
    if type(subjects) is str:
      subjects = int(subjects[len(self.SUBJECT_NAME):])
    if type(subjects) is list and type(subjects[0]) is not int:
      subjects = [int(subject[len(self.SUBJECT_NAME):]) for subject in subjects]

    condition = (type(subjects), type(doses))

    if condition not in self.conditions_dict.keys():
      raise TypeError(f"Unsupported Type combination {condition}!")
    return self.conditions_dict[condition](subjects, doses, **kwargs)

  def load_one_data(self, subject: int, dose: str, **kwargs):
    return self.load_data_by_subject(subject, [dose], **kwargs)

  def load_data_by_dose(self, subjects: list[int], dose: str, **kwargs):
    arr = []
    for subject in subjects:
      filepath = os.path.join(self.datadir, f'{self.SUBJECT_NAME}{subject}',
                              f'{self.SUBJECT_NAME}{subject}_{dose}.npy')
      self.npy_load(filepath, **kwargs)
      arr.append(self._data)
    return np.concatenate(arr)

  def load_data_by_subject(self, subject: int, doses: list[str], **kwargs):
    arr = []
    for dose in doses:
      filepath = os.path.join(self.datadir, f'{self.SUBJECT_NAME}{subject}',
                              f'{self.SUBJECT_NAME}{subject}_{dose}.npy')
      self.npy_load(filepath, **kwargs)
      arr.append(self._data)
    return np.concatenate(arr)

  def load_data(self, subjects: Union[int, str, list],
                doses: Union[str, list],
                **kwargs):
    """
    support 3 ways to load data
    :param subjects:
    :param doses:
    :return: data
    """
    return self._load_data(subjects, doses, **kwargs)

  def npy_load(self, filepath, **kwargs):
    self.current_filepath = filepath
    self._data = np.load(filepath)
    console.supplement(f'Loaded `{os.path.split(filepath)[-1]}`', level=2)
    return self._npy_load(**kwargs)

  def pre_process(self,
                  norm=None, shape=None,
                  raw=False, clip=None,
                  ret_norm=False, **kwargs):
    if shape is not None:
      self._data = norm_size(self._data, shape)
    if clip is not None:
      self._data = np.clip(self._data, clip[0], clip[1])
    self._pre_process(**kwargs)
    if not raw:
      self._data = normalize(self._data, norm, ret_norm=ret_norm)



if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  reader = NpyReader(filePath)
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
