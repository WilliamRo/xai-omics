import os
import numpy as np
from matplotlib import pyplot as plt

from tframe import console


# npy data load function
def load_numpy_data(datadir: str, subjects, doses):
  if type(subjects) is int and type(doses) in [list, np.ndarray]:
    return load_data_by_subject(datadir, subjects, doses)
  elif type(subjects) in [list, np.ndarray] and type(doses) is str:
    return load_data_by_dose(datadir, subjects, doses)
  elif type(subjects) is int and type(doses) is str:
    filepath = os.path.join(datadir, f'subject{subjects}',
                            f'subject{subjects}_{doses}.npy')
    return np.load(filepath)
  else:
    raise TypeError("Subjects or Doses type is wrong!")


def load_data_by_dose(datadir: str, subjects: list, dose: str):
  arr = []
  for subject in subjects:
    if type(subject) in [str, np.str_]:
      subject = int(subject[7:])
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(np.load(filepath))
    console.supplement(f'Loaded `{filepath}`', level=2)
  return np.concatenate(arr)


def load_data_by_subject(datadir: str, subject: int, doses: list):
  arr = []
  if type(subject) is str:
    subject = int(subject[7:])
  for dose in doses:
    filepath = os.path.join(datadir, f'subject{subject}',
                            f'subject{subject}_{dose}.npy')
    arr.append(np.load(filepath))
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
  if th.use_color:
    data = get_color_data(data, "rainbow")
    # data = get_color_data(data, "inferno")
  if th.use_tanh:
    k = 10.
    data = np.tanh(k * data)
  return data


# npy end
def get_color_data(data, cmap):
  sm = plt.colormaps[cmap]
  cm = sm(data[:, ..., 0])[:, ..., :-1]
  return cm




if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  img = load_numpy_data(filePath, 1, ['Full', '1-4'])

  print(get_color_data(img, "rainbow"))
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
