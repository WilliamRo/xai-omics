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


def normalize(arr, norm=None):
  if norm is None:
    norm = np.max(arr)
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
  if np.all(data_shape >= shape):
    for i in range(num):
      if data_shape[i] == shape[i]:
        continue
      else:
        if data_shape[i] % 2 != 0:
          data = np.delete(data, 0, axis=i)
        cut = (data.shape[i] - shape[i]) // 2
        data = np.delete(data, np.s_[0:cut], axis=i)
        data = np.delete(data, np.s_[data.shape[i]-cut:data.shape[i]], axis=i)
  else:
    raise ValueError('The normalized shape is too large!')
  return data


def pre_process(data, tags=None, norm=None,
                use_suv=False, clip=None, cmap=None, shape=None):
  if shape is not None:
    data = norm_size(data, shape)
  if use_suv:
    data = calc_SUV(data, tags)
  if clip is not None:
    data = np.clip(data, clip[0], clip[1])
  if cmap is not None:
    data = get_color_data(data, cmap)
  return normalize(data, norm)
  # return data
