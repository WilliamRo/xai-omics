from typing import Union

import numpy as np


def dicom_time(t):
  t = str(int(t))
  if len(t) == 5:
    t = '0' + t
  h_t = float(t[0:2])
  m_t = float(t[2:4])
  s_t = float(t[4:6])
  return h_t * 3600 + m_t * 60 + s_t


def calc_SUV(data, tags=None, norm=False, **kwargs):
  if tags is not None:
    decay_time = dicom_time(tags['ST']) - dicom_time(tags['RST'])
    decay_dose = float(tags['RTD']) * pow(2, -float(decay_time) / float(tags['RHL']))
    SUVbwScaleFactor = (1000 * float(tags['PW'])) / decay_dose
  else:
    SUVbwScaleFactor = get_suv_factor(**kwargs)
  if norm:
    PET_SUV = (data * float(tags['RS']) + float(tags['RI'])) * SUVbwScaleFactor
  else:
    PET_SUV = data * SUVbwScaleFactor
  return PET_SUV


def get_suv_factor(decay_dose: float, weight):
  return 1000 * weight / decay_dose


def normalize(arr: np.ndarray, norm=None, ret_norm=False, margin: Union[tuple, list]=None):
  if margin is not None:
    assert len(margin) == len(arr.shape)
    norm_arr = arr
    for i in range(len(margin)):
      if margin[i] == 0: continue
      norm_arr = np.delete(norm_arr, np.s_[0:margin[i]], axis=i, )
      norm_arr = np.delete(norm_arr, np.s_[-margin[i]:], axis=i)
  else:
    norm_arr = arr
  if norm is None:
    norm = np.max(norm_arr)
    if ret_norm:
      return arr / norm, norm
  return arr / norm


def get_color_data(data, cmap):
  import matplotlib.pyplot as plt
  sm = plt.colormaps[cmap]
  cm = sm(data[:, ..., 0])[:, ..., :-1]
  return cm


def norm_size(data, shape):
  """
  :param data:
  :param shape: must be even
  :return:
  """
  data_shape = data.shape
  shape = tuple(shape)
  assert len(data_shape) == len(shape)
  num = len(shape)
  for i in range(num):
    if data_shape[i] == shape[i]:
      continue
    elif data_shape[i] < shape[i]:
      zero_add = shape[i] - data_shape[i]
      tmp = list(data.shape)
      tmp[i] = zero_add
      data = np.append(data, np.zeros(tmp), axis=i)
    elif data_shape[i] > shape[i]:
      if data_shape[i] % 2 != 0:
        data = np.delete(data, 0, axis=i)
      cut = (data.shape[i] - shape[i]) // 2
      data = np.delete(data, np.s_[0:cut], axis=i)
      data = np.delete(data, np.s_[data.shape[i] - cut:data.shape[i]], axis=i)

  return data


