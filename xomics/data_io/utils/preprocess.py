import numpy as np


def dicom_time(t):
  t = str(int(t))
  if len(t) == 5:
    t = '0' + t
  h_t = float(t[0:2])
  m_t = float(t[2:4])
  s_t = float(t[4:6])
  return h_t * 3600 + m_t * 60 + s_t


def calc_SUV(data, tags, norm=False):
  decay_time = dicom_time(tags['ST']) - dicom_time(tags['RST'])
  decay_dose = float(tags['RTD']) * pow(2, -float(decay_time) / float(tags['RHL']))
  SUVbwScaleFactor = (1000 * float(tags['PW'])) / decay_dose
  if norm:
    PET_SUV = (data * float(tags['RS']) + float(tags['RI'])) * SUVbwScaleFactor
  else:
    PET_SUV = data * SUVbwScaleFactor
  return PET_SUV


def normalize(arr, norm=None, ret_norm=False):
  if norm is None:
    norm = np.max(arr)
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
  assert np.all(data_shape <= shape)
  if np.any(data_shape < shape):
    for i in range(num):
      if data_shape[i] == shape[i]:
        continue
      else:
        zero_add = shape[i] - data_shape[i]
        tmp = list(data.shape)
        tmp[i] = zero_add
        data = np.append(data, np.zeros(tmp), axis=i)
  return data


