import os
import numpy as np
from pydicom import dcmread
import SimpleITK as sitk




def rd_file(filepath):
  """
  use simpleITK to read file
  :param filepath:
  :return: data array
  """
  itk_img = sitk.ReadImage(filepath)
  img = sitk.GetArrayFromImage(itk_img)
  return img


def rd_series(dirpath):
  """
  use SimpleITK to read image series
  :param dirpath: directory path name
  :return:
  """
  series_reader = sitk.ImageSeriesReader()
  filenames = series_reader.GetGDCMSeriesFileNames(dirpath)
  series_reader.SetFileNames(filenames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  return images


def wr_file(arr, pathname):
  sitk.WriteImage(arr, pathname)


def get_tags(dirpath, suv=True, **kwargs):
  if suv:
    return get_suv_tags(dirpath, **kwargs)
  else:
    return get_all_tags(dirpath, **kwargs)

# todo: get all tags from CT
def get_all_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = dcmread(dirpath)
  elements = data.elements()
  tags = list(elements)
  return tags


def get_suv_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = dcmread(dirpath)
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


def wr_tags(tags, path):
  with open(path, 'w+') as f:
    for name, val in tags.items():
      f.write(f'{name},{val}\n')


def npy_save(data, filepath):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  np.save(filepath, data)




if __name__ == '__main__':
  pass
  # tags = get_tags(os.path.join(path, s))
  # wr_tags(tags, path+'tags.txt')

