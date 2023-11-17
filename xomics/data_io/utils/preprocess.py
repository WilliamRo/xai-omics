from typing import Union

import numpy as np


def dicom_time(t):
  t = str(int(float(t)))
  if len(t) == 5:
    t = '0' + t
  h_t = float(t[0:2])
  m_t = float(t[2:4])
  s_t = float(t[4:6])
  return h_t * 3600 + m_t * 60 + s_t


def calc_SUV(data, tags=None, norm=False, advance=False, **kwargs):
  if advance:
    return calc_SUV_advance(data, tags, norm)
  if 'DD' not in tags.keys():
    decay_time = dicom_time(tags['ST']) - dicom_time(tags['RST'])
    decay_dose = float(tags['RTD']) * pow(2, -float(decay_time) / float(tags['RHL']))
    SUVbwScaleFactor = (1000 * float(tags['PW'])) / decay_dose
  else:
    SUVbwScaleFactor = 1000 * float(tags['PW']) / tags['DD']
  if norm:
    PET_SUV = (data * float(tags['RS']) + float(tags['RI'])) * SUVbwScaleFactor
  else:
    PET_SUV = data * SUVbwScaleFactor
  return PET_SUV


def reverse_suv(data, tags):
  SUVbwScaleFactor, RS, RI = get_suv_factor(tags)
  raw_data = data / SUVbwScaleFactor
  return raw_data


def calc_SUV_advance(data, tags, norm=False):
  SUVbwScaleFactor, RS, RI = get_suv_factor(tags)
  if norm:
    PET_SUV = (data * float(RS) + float(RI)) * SUVbwScaleFactor
  else:
    PET_SUV = data * SUVbwScaleFactor
  return PET_SUV


def get_suv_factor(tags):
  ST = tags['SeriesTime']
  RIS = tags['RadiopharmaceuticalInformationSequence'][0]
  RST = str(RIS['RadiopharmaceuticalStartTime'].value)
  RTD = str(RIS['RadionuclideTotalDose'].value)
  RHL = str(RIS['RadionuclideHalfLife'].value)
  PW = tags['PatientWeight']
  RS = tags['RescaleSlope']
  RI = tags['RescaleIntercept']

  decay_time = dicom_time(ST) - dicom_time(RST)
  decay_dose = float(RTD) * pow(2, -float(decay_time) / float(RHL))
  SUVbwScaleFactor = (1000 * float(PW)) / decay_dose

  return SUVbwScaleFactor, RS, RI


def normalize(arr: np.ndarray, norm=None, ret_norm=False):
  if norm is None:
    max_data = np.max(arr)
    min_data = np.min(arr)
    norm = [max_data, min_data]
    if ret_norm:
      return (arr - norm[1]) / (norm[0] - norm[1]), norm
  return (arr - norm[1]) / (norm[0] - norm[1])


def crop_by_margin(arr, margin: Union[tuple, list]):
  assert len(margin) == len(arr.shape)
  for i in range(len(margin)):
    if margin[i] == 0:
      continue
    arr = np.delete(arr, np.s_[0:margin[i]], axis=i)
    arr = np.delete(arr, np.s_[-margin[i]:], axis=i)
  return arr


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


