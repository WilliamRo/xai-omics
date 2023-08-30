import os
import numpy as np
from matplotlib import pyplot as plt

from tframe import console
from xomics.data_io.raw_reader import calc_SUV

SUBJECT_NAME = 'subject'

def load_tags(filepath):
  tags = {}
  with open(filepath, 'r') as f:
    for data in f.readlines():
      data = data.split(',')
      tags[data[0]] = float(data[1])
  return tags


def load_numpy_data(datadir: str, subjects, doses, **kwargs):
  if type(subjects) is str:
    subjects = int(subjects[len(SUBJECT_NAME):])
  if type(subjects) is list and type(subjects[0]) is not int:
    subjects = [int(subject[len(SUBJECT_NAME):]) for subject in subjects]

  condition = (type(subjects), type(doses))
  conditions_dict = {
    (int, list): load_data_by_subject,
    (list, str): load_data_by_dose,
    (int, str): load_one_data,
    (list, dict): load_data_pair,
  }
  if condition not in conditions_dict.keys():
    raise TypeError(f"Unsupported Type combination {condition}!")

  return conditions_dict[condition](datadir, subjects, doses, **kwargs)


def load_one_data(datadir: str, subject: int, dose: str, **kwargs):
  return load_data_by_subject(datadir, subject, [dose], **kwargs)


def load_data_by_dose(datadir: str, subjects: list[int], dose: str, **kwargs):
  arr = []
  for subject in subjects:
    filepath = os.path.join(datadir, f'{SUBJECT_NAME}{subject}',
                            f'{SUBJECT_NAME}{subject}_{dose}.npy')
    data, _ = npy_load(filepath, **kwargs)
    arr.append(data)
  return np.concatenate(arr)


def load_data_by_subject(datadir: str, subject: int, doses: list[str], **kwargs):
  arr = []
  for dose in doses:
    filepath = os.path.join(datadir, f'{SUBJECT_NAME}{subject}',
                            f'{SUBJECT_NAME}{subject}_{dose}.npy')
    data, _ = npy_load(filepath, **kwargs)
    arr.append(data)
  return np.concatenate(arr)


def load_data_pair(datadir: str, subjects: list[int], doses: dict, **kwargs):
  features = []
  targets = []
  for subject in subjects:
    f_filepath = os.path.join(datadir, f'{SUBJECT_NAME}{subject}',
                              f'{SUBJECT_NAME}{subject}_{doses["feature"]}.npy')
    t_filepath = os.path.join(datadir, f'{SUBJECT_NAME}{subject}',
                              f'{SUBJECT_NAME}{subject}_{doses["target"]}.npy')

    feature, norm = npy_load(f_filepath, **kwargs)
    target = npy_load(t_filepath, norm=norm, **kwargs)

    features.append(feature)
    targets.append(target)

  return np.concatenate(features), np.concatenate(targets)


def normalize(arr, norm=None):
  if norm is None:
    norm = np.max(arr)
    return arr / norm, norm
  return arr / norm


def load_data(datadir: str,
              subjects: int | str | list,
              doses: str | list | dict,
              **kwargs):
  """
  support 3 ways to load data
  :param datadir:  data file directory
  :param subjects:
  :param doses:
  :return: data
  """

  data = load_numpy_data(datadir, subjects, doses, **kwargs)

  return data


def get_color_data(data, cmap):
  sm = plt.colormaps[cmap]
  cm = sm(data[:, ..., 0])[:, ..., :-1]
  return cm


def npy_load(filepath, norm=None, use_suv=False, **kwargs):
  data = np.load(filepath)
  console.supplement(f'Loaded `{os.path.split(filepath)[-1]}`', level=2)
  if use_suv:
    tmp = os.path.split(filepath)
    tagpath = os.path.join(tmp[0], f'tags_{tmp[1][:-3]}txt')
    tags = load_tags(tagpath)
    return pre_process(data, tags=tags, norm=norm, use_suv=use_suv, **kwargs)
  return pre_process(data, norm=norm, **kwargs)


def norm_size(data, shape):
  """
  :param data:
  :param shape: must be even
  :return:
  """
  data_shape = data.shape
  shape = tuple(shape)
  assert len(data_shape) == len(shape)
  num = len(shape)
  if np.all(data_shape >= shape):
    for i in range(num):
      if data_shape[i] == shape[i]:
        continue
      else:
        if data_shape[i] % 2 != 0:
          data = np.delete(data, 0, axis=i)
        cut = (data.shape[i] - shape[i]) // 2
        data = np.delete(data, np.s_[0:cut], axis=i)
        data = np.delete(data, np.s_[data.shape[i]-cut:data.shape[i]], axis=i)
  else:
    raise ValueError('The normalized shape is too large!')
  return data


def pre_process(data, tags=None, norm=None,
                use_suv=False, clip=None, cmap=None, shape=None):
  if shape is not None:
    data = norm_size(data, shape)
  if use_suv:
    data = calc_SUV(data, tags)
  if clip is not None:
    data = np.clip(data, clip[0], clip[1])
  if cmap is not None:
    data = get_color_data(data, cmap)
  return normalize(data, norm)
  # return data



if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  img = load_data(filePath, 2, ['Full', '1-2'], shape=[1, 608, 440, 440, 1])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
