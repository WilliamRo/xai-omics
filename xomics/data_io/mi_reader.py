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
  if type(subjects) is int and type(doses) in [list, np.ndarray]:
    return load_data_by_subject(datadir, subjects, doses)
  elif type(subjects) in [list, np.ndarray] and type(doses) is str:
    return load_data_by_dose(datadir, subjects, doses)
  elif type(subjects) is int and type(doses) is str:
    filepath = os.path.join(datadir, f'subject{subjects}',
                            f'subject{subjects}_{doses}.npy')
    return npy_load(filepath)
  else:
    raise TypeError("Subjects or Doses type is wrong!")


def load_data_by_dose(datadir: str, subjects: list, dose: str):
  arr = []
  for subject in subjects:
    if type(subject) in [str, np.str_]:
      subject = int(subject[7:])
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(npy_load(filepath))
    console.supplement(f'Loaded `{filepath}`', level=2)
  return np.concatenate(arr)


def load_data_by_subject(datadir: str, subject: int, doses: list):
  arr = []
  if type(subject) is str:
    subject = int(subject[7:])
  for dose in doses:
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(npy_load(filepath))
    console.supplement(f'Loaded `{filepath}`', level=2)
  return np.concatenate(arr)


def load_data(datadir: str,
              subjects: int | str | list | np.ndarray,
              doses: str | list | np.ndarray):
  """
  support 3 ways to load data
  :param datadir:  data file directory
  :param subjects:
  :param doses:
  :return:
  """
  from uld_core import th

  data = load_numpy_data(datadir, subjects, doses)

  if th.use_clip != np.Inf:
    data = np.clip(data, 0, th.use_clip) / th.use_clip
  if th.use_color:
    data = get_color_data(data, "nipy_spectral")
    # data = get_color_data(data, "inferno")
  # if th.use_tanh > 0:
  #   k = th.use_tanh
  #   data = np.tanh(k * data)
  return data


# npy end
def get_color_data(data, cmap):
  sm = plt.colormaps[cmap]
  cm = sm(data[:, ..., 0])[:, ..., :-1]
  return cm


def npy_load(filepath, slice_num=608, suv=True):
  data = np.load(filepath)
  if suv:
    tmp = os.path.split(filepath)
    tagpath = os.path.join(tmp[0], f'tags_{tmp[1][:-3]}txt')
    tags = load_tags(tagpath)
    data = calc_SUV(data, tags)
    # print(np.max(suv), np.mean(suv))

  if data.shape[1] % 2 != 0:
    data = data[:, 1:]
  cut = (data.shape[1] - slice_num) // 2
  data = data[:, cut:-cut]
  return data




if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  img = load_numpy_data(filePath, 1, ['Full'])

  print(img.shape)
  # keys = ['Full_dose',
  #         '1-2 dose',
  #         '1-4 dose',
  #         '1-10 dose',
  #         '1-20 dose',
  #         '1-50 dose',
  #         '1-100 dose',
  #         ]
  # subjects = os.listdir('../../data/01-ULD/')
  # datapath = '../../data/01-ULD'
  # path = '../../data/'
  # num = 1
  # for subject in subjects:
  #   for dose in keys:
  #     results = rd_subject(datapath, subject, dose)
  #     cnum = num
  #     for i in range(results.shape[0]):
  #       filepath = os.path.join(path, f'subject{cnum}')
  #       if not os.path.exists(filepath):
  #         os.mkdir(filepath)
  #       np.save(os.path.join(filepath,f'subject{cnum}_{dose[:-5]}.npy'),results[i:i+1])
  #       print(f'subject{cnum} {dose} completed!')
  #       cnum+=1
  #   num=cnum
