import pydicom
import os
import gzip
import numpy as np

from tools import data_processing
from medpy.io import load, save
from tqdm import tqdm


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
  input_BA_dir = r'D:\Gordon\project_for_learning\xai-alfa\70-ANALYST3D\data\BA & MIA\radioimage\BA'
  save_path_structure = r'D:\Gordon\project_for_learning\xai-alfa\70-ANALYST3D\data\npy_data\BA_structure.npy'
  save_path_mask = r'D:\Gordon\project_for_learning\xai-alfa\70-ANALYST3D\data\npy_data\BA_mask.npy'
  wc = -600.0
  ww = 1200.0

  mask_BA = [file for file in os.listdir(input_BA_dir) if '.nii.gz' in file]
  marked_BA = [[file.split('-')[0], file.split('-')[1].split('.')[0]]
               for file in mask_BA]

  # Get the structure
  # The shape of the datas is [Patient, Slice, H, W]
  datas = []
  for dir in tqdm(marked_BA):
    all_dir = os.path.join(input_BA_dir, dir[0], dir[1])
    all_file = os.listdir(all_dir)

    single_data = []
    rescale_intercept, rescale_slope = [], []
    for file in all_file:
      reading_data = pydicom.dcmread(os.path.join(all_dir, file))
      single_data.append(reading_data.pixel_array)
      rescale_intercept.append(reading_data.RescaleIntercept)
      rescale_slope.append(reading_data.RescaleSlope)

    ri = (float(rescale_intercept[0])
          if are_all_elements_same(rescale_intercept) else None)
    rs = (float(rescale_slope[0])
          if are_all_elements_same(rescale_slope) else None)

    assert rescale_intercept is not None
    assert rescale_slope is not None

    single_data = np.array(single_data)[::-1]

    datas.append(pixel_correction_and_normalization(single_data, wc, ww,
                                                    ri, rs))

  data_processing.save_as_npy(save_path_structure, datas)

  # Get the mask
  mask_dir = os.path.join(input_BA_dir, 'uncompress')
  mask_BA = os.listdir(mask_dir)
  masks = [read_nii_with_medpy(os.path.join(mask_dir, mask_file))
           for mask_file in mask_BA]

  data_processing.save_as_npy(save_path_mask, masks)




  print()