from roma import finder

import joblib
import numpy as np
import os
import pydicom





def rd_file(filepath):
  """
  use simpleITK to read file
  :param filepath:
  :return: data array
  """
  import SimpleITK as sitk

  itk_img = sitk.ReadImage(filepath)
  img = sitk.GetArrayFromImage(itk_img)
  return img


def rd_series(dirpath):
  """
  use SimpleITK to read image series
  :param dirpath: directory path name
  :return:
  """
  file_paths = finder.walk(dirpath)
  dcms = [pydicom.dcmread(path)
          for path in file_paths]
  dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
  data = [d.pixel_array for d in dcms]
  return np.stack(data, axis=0)


def wr_file(arr, pathname):
  import SimpleITK as sitk
  sitk.WriteImage(arr, pathname)


def get_tags(dirpath, suv=True, **kwargs):
  if suv:
    return get_suv_tags(dirpath, **kwargs)
  else:
    return get_all_tags(dirpath, **kwargs)


def get_all_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = pydicom.dcmread(dirpath)
  elements = data.elements()
  tags = list(elements)
  return tags


def get_suv_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = pydicom.dcmread(dirpath)
  ST = data.SeriesTime
  AT = data.AcquisitionTime
  PW = data.PatientWeight
  RIS = data.RadiopharmaceuticalInformationSequence[0]
  RST = str(RIS['RadiopharmaceuticalStartTime'].value)
  RTD = str(RIS['RadionuclideTotalDose'].value)
  RHL = str(RIS['RadionuclideHalfLife'].value)
  RS = data.RescaleSlope
  RI = data.RescaleIntercept
  dcm_tag = {
    'ST': ST,
    'AT': AT,
    'PW': PW,
    'RST': RST,
    'RTD': RTD,
    'RHL': RHL,
    'RS': RS,
    'RI': RI
  }
  # for name, key in dcm_tag.items():
  #   print(name, key)
  return dcm_tag


def wr_tags(tags, path, suv=False):
  if suv:
    wr_suv_tags(tags, path)
  else:
    joblib.dump(tags, path)


def wr_suv_tags(tags, path):
  with open(path, 'w+') as f:
    for name, val in tags.items():
      f.write(f'{name},{val}\n')


def rd_tags(path):
  return joblib.load(path)


def npy_save(data, filepath):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  np.save(filepath, data)




if __name__ == '__main__':
  pass
  # tags = get_tags(os.path.join(path, s))
  # wr_tags(tags, path+'tags.txt')

