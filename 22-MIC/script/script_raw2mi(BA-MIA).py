import pydicom
import os
import gzip
import numpy as np

from tools import data_processing
from medpy.io import load, save
from tqdm import tqdm
from xomics import MedicalImage


def pixel_correction_and_normalization(input_array, wc, ww, ri, rs):
  '''
  input_array: np.ndarray (slice, H, W)
  wc: Window Center
  ww: Window Width
  ri: Rescale Intercept
  rs: Rescale Slope
  '''
  array = np.array(input_array, dtype=np.int16) * rs + ri
  window_min, window_max = wc - ww / 2.0, wc + ww / 2.0
  array[array < window_min] = window_min
  array[array > window_max] = window_max

  return (array - np.min(array)) / (np.max(array) - np.min(array))


def pixel_correction(input_array, ri, rs):
  '''
  input_array: np.ndarray (slice, H, W)
  ri: Rescale Intercept
  rs: Rescale Slope
  '''
  return np.array(input_array, dtype=np.int16) * rs + ri


def are_all_elements_same(input_list):
  if not input_list:
    return True

  first_element = input_list[0]
  for element in input_list[1:]:
    if element != first_element:
      return False
  return True


def read_nii_with_medpy(file):
  '''
  input: file path of nii file
  output: datas in numpy
          raw datas shape: (length, width, slice) (512, 512, 97)
          transformed datas shape: (slice, length, width) (97, 512, 512)
  '''
  data, header = load(file)

  return np.transpose(data, (2, 1, 0))


if __name__ == '__main__':
  input_dir = r'../../data/03-BA & MIA/radioimage'
  input_types = ['BA', 'MIA']

  for input_type in input_types:
    ct_dir = os.path.join(input_dir, input_type)
    label_dir = os.path.join(ct_dir, 'labels')
    save_dir = os.path.join('../../data/03-BA & MIA/mi', input_type)

    wc = -600.0
    ww = 1200.0

    label_names = os.listdir(label_dir)

    # Get the structure and mask
    # The shape of the datas is [Patient, Slice, H, W]
    for name in tqdm(label_names):
      # if name not in ['2644420-8538069.nii', '7579614-7946379.nii']:
      #   continue

      dmc_dir = os.path.join(
        ct_dir, name.split('-')[0], name.split('-')[1].split('.')[0])
      dcm_file = os.listdir(dmc_dir)

      data = []
      rescale_intercept, rescale_slope = [], []
      for file in dcm_file:
        reading_data = pydicom.dcmread(os.path.join(dmc_dir, file))
        data.append(reading_data.pixel_array)
        rescale_intercept.append(reading_data.RescaleIntercept)
        rescale_slope.append(reading_data.RescaleSlope)

      ri = (float(rescale_intercept[0])
            if are_all_elements_same(rescale_intercept) else None)
      rs = (float(rescale_slope[0])
            if are_all_elements_same(rescale_slope) else None)

      assert rescale_intercept is not None
      assert rescale_slope is not None

      data = np.array(data)[::-1]
      data = pixel_correction(input_array=data, ri=ri, rs=rs)
      data = np.array(data, dtype=np.float16)
      assert not np.any(np.isinf(data))

      label = read_nii_with_medpy(os.path.join(label_dir, name))
      label = np.int8(label)
      assert not np.any(np.isinf(label))

      mi: MedicalImage = MedicalImage(
        images={'ct': data}, labels={'label-0': label}, key=name[:-4])

      mi.save(os.path.join(save_dir, name[:-4] + '.mi'))