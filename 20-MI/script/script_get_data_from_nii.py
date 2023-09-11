import nibabel as nib
import numpy as np
import os

from medpy.io import load, save
from tqdm import tqdm



def read_nii_with_medpy(file):
  '''
  input: file path of nii file
  output: datas in numpy
          raw datas shape: (length, width, slice) (512, 512, 97)
          transformed datas shape: (slice, length, width) (97, 512, 512)
  '''
  data, header = load(file)

  return np.transpose(data, (2, 1, 0))


def read_nii_with_nibabel(file):
  '''
  input: file path of nii file
  output: datas in numpy
          raw datas shape: (length, width, slice) (512, 512, 97)
          transformed datas shape: (slice, length, width) (97, 512, 512)
  '''
  nifti_file = nib.load(file)
  data = nifti_file.get_fdata()
  data = np.array(data)

  return np.transpose(data, (2, 1, 0))


def get_segmentation_data(input_data: np.ndarray, type=None):
  '''
  input: datas in numpy
         datas shape: (slice, length, width) (97, 512, 512)
  output: liver datas or tumor datas
          datas shape: (slice, length, width) (97, 512, 512)
  '''
  if type == 'liver':
    type = 1
  elif type == 'tumor':
    type = 2
  else:
    raise TypeError("The 'type' parameter is incorrect!")
  index = np.where(input_data == type)
  output_data = np.zeros_like(input_data)
  output_data[index] = type

  return output_data


def save_as_npy(file_path, input_list):
  '''
  The input is a list
  example: list = [[1,2,3], [4, 5], [6, 7, 8, 9]]
  '''
  array = np.array(input_list, dtype=object)
  np.save(file_path, array)


def load_npy_file(file_path):
  return np.load(file_path, allow_pickle=True).tolist()



if __name__ == '__main__':
  dir = '../data/LiTS/train_data'
  npy_save_path_targets = '../data/npy_data/LiTS_train_targets.npy'
  npy_save_path_features = '../data/npy_data/LiTS_train_features.npy'

  file_names = os.listdir(dir)
  segmentation_names = [name for name in file_names if 'segmentation' in name]
  volume_names = [name for name in file_names if 'volume' in name]

  segmentation_data, volume_data = [], []

  for name in tqdm(segmentation_names):
    data = read_nii_with_medpy(os.path.join(dir, name))
    segmentation_data.append(data)
  save_as_npy(npy_save_path_targets, segmentation_data)
  segmentation_data = None

  for name in tqdm(volume_names):
    data = read_nii_with_medpy(os.path.join(dir, name))
    volume_data.append(data)

  save_as_npy(npy_save_path_features, volume_data)
  volume_data = None
  print('saving finished!!!')


  print()