from xomics.data_io.utils.raw_rw import rd_file, rd_series, get_tags

import os
import numpy as np


def rd_uld_test(dirpath, name_list, outputs=False):
  """
  read uld test raw data
  :param name_list: number id of data
  :param dirpath:
  :return: data
  """
  if outputs:
    dirpath = os.path.join(dirpath, 'outputs')
  file_list = os.listdir(dirpath)
  if not outputs:
    file_list.remove('seg')
    file_list.remove('outputs')
    file_list.remove('tags.txt')
  files = file_list
  arr = []
  for file in files:
    if outputs:
      tmp = int(file.split('.')[0][3:])
    else:
      tmp = int(file.split('_')[-1].split('.')[0])
    if tmp in name_list:
      filepath = os.path.join(dirpath, file)
      arr.append(rd_file(filepath))
      print(f'...loaded {file}')
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
  patients.sort()
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
