import os

import numpy as np
import SimpleITK as sitk



def normalize(a):
  # for i in range(a.shape[0]):
  #   a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
  a = (a - np.min(a)) / (np.max(a) - np.min(a))
  return a


def rdSeries(dirPath, norm = True):
  series_reader = sitk.ImageSeriesReader()
  fileNames = series_reader.GetGDCMSeriesFileNames(dirPath)
  series_reader.SetFileNames(fileNames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  if norm:
    return normalize(images)
  else:
    return images


def rdSubject(dataPath, subject, dose = "Full_dose", patient_num = 6, norm = True):
  patients = os.listdir(dataPath + subject)[:patient_num]
  images = []

  for patient in patients:
    dirPath = dataPath + subject + f'/{patient}/' + dose
    images.append(rdSeries(dirPath, norm))

  results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  return results



if __name__ == '__main__':
  filePath = '../../data/01-ULD/Subject_372-387/20102022_1_20221020_164152/1' \
             '-2 dose/'
  img = rdSeries(filePath)
  print(img.shape)





