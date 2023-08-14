import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage import exposure
from tframe import console


def normalize(a):
  a = (a - np.min(a)) / np.max(a)
  return a


def rd_series(dirPath, hist_equal=False):
  series_reader = sitk.ImageSeriesReader()
  fileNames = series_reader.GetGDCMSeriesFileNames(dirPath)
  series_reader.SetFileNames(fileNames)
  data = series_reader.Execute()

  images = sitk.GetArrayFromImage(data)
  cut = (images.shape[0] - 608) // 2  # find_num(images.shape[0], 32)
  images = images[cut:-cut]
  # return 1 - normalize(images)
  if hist_equal:
    return exposure.equalize_hist(normalize(images), mask=images != 0)
  else:
    return normalize(images)
    # return images


def output_results(arr, pathname):
  sitk.WriteImage(arr, pathname)


def rd_subject(dataPath, subject, dose="Full_dose",
               patient_num=6, hist_equal=False):
  patients = os.listdir(os.path.join(dataPath, subject))[:patient_num]
  images = []

  for patient in patients:
    dirPath = os.path.join(dataPath, subject, patient, dose)
    images.append(rd_series(dirPath, hist_equal))

  results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  results = results.reshape(results.shape + (1,))

  return results


def rd_data(dataPath, subjects: List, dose="Full_dose",
            patient_num=6, hist_equal=False):
  data = []
  for subject in subjects:
    data.append(rd_subject(dataPath, subject, dose, patient_num, hist_equal))
  data = np.concatenate(data, axis=0)

  return data


# npy data load function
def load_numpy_data(datadir: str, subjects, doses):
  if type(subjects) is int and type(doses) is list:
    return load_data_by_subject(datadir, subjects, doses)
  elif type(subjects) is list and type(doses) is str:
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


def load_data(datadir: str, subjects, doses):
  return load_numpy_data(datadir, subjects, doses)


# npy end


if __name__ == '__main__':
  filePath = '../../data/01-ULD/'
  img = load_numpy_data(filePath, 1, ['Full', '1-4'])
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
