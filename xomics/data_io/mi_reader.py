import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def normalize(a):
  # for i in range(a.shape[0]):
  #   a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
  # print(np.max(a))
  # print(np.count_nonzero(a > 5000))
  # b = np.ndarray.flatten(a)
  # plt.hist(b[b!=0], bins=50, range = (0,6000))
  # plt.show()
  a = np.clip(a, a_min=None, a_max=3500)
  a = (a - np.min(a)) / (np.max(a) - np.min(a))
  return a


def rd_series(dirPath, norm=True):
  series_reader = sitk.ImageSeriesReader()
  fileNames = series_reader.GetGDCMSeriesFileNames(dirPath)
  series_reader.SetFileNames(fileNames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  if norm:
    return normalize(images)
  else:
    return images


def rd_subject(dataPath, subject, dose="Full_dose", patient_num=6, norm=True):
  patients = os.listdir(dataPath + subject)[:patient_num]
  images = []

  for patient in patients:
    dirPath = os.path.join(dataPath, subject, patient, dose)
    images.append(rd_series(dirPath, norm))

  results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  results = results.reshape(results.shape + (1,))

  return results


def rd_data(dataPath, subjects: List, dose="Full_dose", patient_num=6,
            norm=True):
  data = []
  for subject in subjects:
    data.append(rd_subject(dataPath, subject, dose, patient_num, norm))
  data = np.concatenate(data, axis=0)

  return data


if __name__ == '__main__':
  filePath = '../../data/01-ULD/Subject_372-387/20102022_1_20221020_164152/1' \
             '-4 dose/'
  img = rd_series(filePath)
  print(img.shape)
