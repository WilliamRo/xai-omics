import os
import numpy as np
from matplotlib import pyplot as plt

from tframe import console
from xomics.data_io.raw_reader import calc_SUV


def load_tags(filepath):
  tags = {}
  with open(filepath, 'r') as f:
    for data in f.readlines():
      data = data.split(',')
      tags[data[0]] = float(data[1])
  return tags


# npy data load function
def load_numpy_data(datadir: str, subjects, doses):
  if type(subjects) is str:
    subjects = int(subjects[7:])
  if type(subjects) is list:
    subjects = [int(subject[7:]) for subject in subjects]

  condition = (type(subjects), type(doses))
  conditions_dict = {
    (int, list): load_data_by_subject,
    (list, str): load_data_by_dose,
    (int, str): load_one_data,
    (list, dict): load_data_pair,
  }

  return conditions_dict[condition](datadir, subjects, doses)


def load_one_data(datadir: str, subject: int, dose: str):
  return load_data_by_subject(datadir, subject, [dose])


def load_data_by_dose(datadir: str, subjects: list[int], dose: str):
  arr = []
  for subject in subjects:
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(npy_load(filepath))
  return np.concatenate(arr)


def load_data_by_subject(datadir: str, subject: int, doses: list[str]):
  arr = []
  for dose in doses:
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(npy_load(filepath))
  return np.concatenate(arr)


def load_data_pair(datadir: str, subjects: list[int], doses: dict):
  features = []
  targets = []
  for subject in subjects:
    f_filepath = os.path.join(datadir, f'subject{subject}',
                              f'subject{subject}_{doses["feature"]}.npy')
    t_filepath = os.path.join(datadir, f'subject{subject}',
                              f'subject{subject}_{doses["target"]}.npy')

    feature = npy_load(f_filepath)
    target = npy_load(t_filepath)
    feature, norm = normalize(feature)
    target = normalize(target, norm)

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
              doses: str | list | dict):
  """
  support 3 ways to load data
  :param datadir:  data file directory
  :param subjects:
  :param doses:
  :return: data
  """
  from uld_core import th

  if th.norm_by_feature:
    return load_numpy_data(datadir, subjects, doses)
  else:
    data = load_numpy_data(datadir, subjects, doses)


  return data


def get_color_data(data, cmap):
  sm = plt.colormaps[cmap]
  cm = sm(data[:, ..., 0])[:, ..., :-1]
  return cm


def npy_load(filepath):
  from uld_core import th
  data = np.load(filepath)
  console.supplement(f'Loaded `{os.path.split(filepath)[-1]}`', level=2)
  if th.use_suv:
    tmp = os.path.split(filepath)
    tagpath = os.path.join(tmp[0], f'tags_{tmp[1][:-3]}txt')
    tags = load_tags(tagpath)
    return pre_process(data, tags)
  return pre_process(data)


def pre_process(data, tags=None):
  from uld_core import th

  if data.shape[1] % 2 != 0:
    data = data[:, 1:]
  cut = (data.shape[1] - th.slice_num) // 2
  data = data[:, cut:-cut]

  if th.use_suv:
    data = calc_SUV(data, tags)
  if th.use_clip != np.Inf:
    data = np.clip(data, 0, th.use_clip) / th.use_clip
  if th.use_color:
    data = get_color_data(data, "nipy_spectral")
  return data




if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  # img = load_numpy_data(filePath, 8, ['Full'])
  img = load_data(filePath, 2, ['Full', '1-2'])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
