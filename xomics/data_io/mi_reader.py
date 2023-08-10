import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage import exposure


def normalize(a):
  a = (a - np.min(a)) / (np.max(a) - np.min(a))
  return a


def rd_series(dirPath):
  series_reader = sitk.ImageSeriesReader()
  fileNames = series_reader.GetGDCMSeriesFileNames(dirPath)
  series_reader.SetFileNames(fileNames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  cut = (images.shape[0] - 608) // 2# find_num(images.shape[0], 32)
  images = images[cut:-cut]
  return normalize(images)


def output_results(arr, pathname):
  sitk.WriteImage(arr, pathname)


def rd_subject(dataPath, subject, dose="Full_dose", patient_num=6):
  patients = os.listdir(dataPath + subject)[:patient_num]
  images = []

  for patient in patients:
    dirPath = os.path.join(dataPath, subject, patient, dose)
    images.append(rd_series(dirPath))

  results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  results = results.reshape(results.shape + (1,))

  return results


def rd_data(dataPath, subjects: List, dose="Full_dose", patient_num=6):
  data = []
  for subject in subjects:
    data.append(rd_subject(dataPath, subject, dose, patient_num))
  data = np.concatenate(data, axis=0)

  return data


if __name__ == '__main__':
  filePath = '../../data/01-ULD/Subject_372-387/20102022_1_20221020_164152/1' \
             '-4 dose/'
  img = rd_series(filePath)#, norm=False)
  print(img.shape)
