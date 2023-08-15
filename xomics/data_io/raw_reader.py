import os
from typing import List
import numpy as np
import SimpleITK as sitk
from skimage import exposure


def normalize(a):
  a = (a - np.min(a)) / np.max(a)
  return a


def rd_file(filepath):
  itk_img = sitk.ReadImage(filepath)
  img = sitk.GetArrayFromImage(itk_img)
  return img


def rd_series(dirpath):
  series_reader = sitk.ImageSeriesReader()
  filenames = series_reader.GetGDCMSeriesFileNames(dirpath)
  series_reader.SetFileNames(filenames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  return images


def wr_file(arr, pathname):
  sitk.WriteImage(arr, pathname)


# for uld train raw data
def rd_uld_train(datapath, subject, dose="Full_dose"):
  patients = str(os.listdir(os.path.join(datapath, subject)))
  images = []

  for patient in patients:
    dirpath = os.path.join(datapath, subject, patient, dose)
    img = normalize(rd_series(dirpath))

    if img.shape[0] % 2 != 0:
      img = img[1:]
    cut = (img.shape[0] - 608) // 2
    img = img[cut:-cut]

    images.append(img)

  results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  results = results.reshape(results.shape + (1,))

  return results


# for uld test raw data
def rd_uld_test(dirpath, datanum=1):
  files = os.listdir(dirpath)[:datanum]
  arr = []
  for file in files:
    filepath = os.path.join(dirpath, file)
    arr.append(rd_file(filepath))
  results = np.stack(arr)
  return results




if __name__ == '__main__':
  path = "../../data/01-ULD/testset/"
  imgs = rd_uld_test(path,2)
  print(imgs.shape)

