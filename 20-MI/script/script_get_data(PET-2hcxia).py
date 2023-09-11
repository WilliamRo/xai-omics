import pydicom
import os
import gzip
import numpy as np

from tools import data_processing
from medpy.io import load, save
from tqdm import tqdm



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
  ct_dir = r'..\..\data\PET\2hcxia\ct'
  pet_dir = r'..\..\data\PET\2hcxia\pet'
  mask_file = r'E:\xai-omics\data\PET\2hcxia\R1_Rel_thres41.0.uint16.nii'
  save_path_ct = r'..\data\npy_data\2hcxia_ct.npy'
  save_path_pet = r'..\data\npy_data\2hcxia_pet.npy'
  save_path_mask = r'..\data\npy_data\2hcxia_mask.npy'

  wc = 40.0
  ww = 300.0

  # Get the ct data
  ct_file = os.listdir(ct_dir)
  single_data = []
  rescale_intercept, rescale_slope = [], []
  for file in ct_file:
    reading_data = pydicom.dcmread(os.path.join(ct_dir, file))
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
  data_ct = [np.array(single_data)]

  data_processing.save_as_npy(save_path_ct, data_ct)

  # Get the pet data
  pet_file = os.listdir(pet_dir)
  data = [pydicom.dcmread(os.path.join(pet_dir, file)).pixel_array
          for file in pet_file]
  data = [np.array(data)[::-1]]

  data_processing.save_as_npy(save_path_pet, data)

  # Get the mask
  masks = [read_nii_with_medpy(os.path.join(mask_file, mask_file))]

  data_processing.save_as_npy(save_path_mask, masks)

