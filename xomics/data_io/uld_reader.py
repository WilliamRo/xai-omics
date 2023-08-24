import os
import numpy as np
from xomics.data_io.raw_reader import rd_file, rd_series, get_tags


def rd_uld_test(dirpath, datanum=1):
  """
  read uld test raw data
  :param dirpath:
  :param datanum: how many data to read
  :return: data
  """
  files = os.listdir(dirpath)[:datanum]
  arr = []
  for file in files:
    filepath = os.path.join(dirpath, file)
    arr.append(rd_file(filepath))
  results = np.stack(arr)
  return results


def rd_uld_train(datapath: str, subject, dose="Full_dose"):
  """
  for uld train raw data
  :param datapath:
  :param subject:
  :param dose:
  :return:
  """
  patients = os.listdir(os.path.join(datapath, subject))
  images = []
  tags = []

  for patient in patients:
    dirpath = os.path.join(datapath, subject, patient, dose)
    img = rd_series(dirpath)

    tag_dict = get_tags(dirpath, isSeries=True)
    img = img.reshape((1,) + img.shape + (1,))

    images.append(img)
    tags.append(tag_dict)

  return images, tags
