from xomics.data_io.utils.raw_rw import rd_file, rd_series, get_tags

import os
import numpy as np


def rd_uld_test(dirpath, name_list):
  """
  read uld test raw data
  :param name_list: number id of data
  :param dirpath:
  :return: data
  """
  file_list = os.listdir(dirpath)
  file_list.remove('seg')
  files = file_list
  arr = []
  for file in files:
    tmp = int(file.split('_')[-1].split('.')[0])
    if tmp in name_list:
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
