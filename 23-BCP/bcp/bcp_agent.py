import random
import os
import numpy as np

from bcp.bcp_set import BCPSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from scipy.ndimage import label, generate_binary_structure


class BCPAgent(DataAgent):

  TFD_FILE_NAME = 'bcp.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from bcp_core import th

    bcp_set = cls.load_as_tframe_data(th.data_dir)

    if len(bcp_set) == 1:
      # If there is only 1 example, we make training set, validation set
      # and testing set the same.
      datasets = bcp_set, bcp_set, bcp_set
    else:
      ratio = [int(x) for x in th.ratio_of_dataset.split(':')]
      assert len(ratio) == 3
      parts = ratio_to_realnum(ratio=ratio, total_num=len(bcp_set))

      datasets = bcp_set.split(
        parts[0], parts[1], parts[2], names=['TrainSet', 'ValSet', 'TestSet'])
    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> BCPSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return BCPSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = BCPSet(data_dict=image_dict, name='BCPSet')


    # Show status
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving datas set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    # Wrap and return

    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir) -> OrderedDict:
    '''
      features and targets are exactly the same.
      In this dataset, the shape of pet data is (175, 440, 440), which is
      about 130 MB
    '''
    print(data_dir)
    image_dict = OrderedDict()

    dir = os.path.abspath(
      os.path.join(data_dir, '../../data/04-Brain-CT-PET/mi'))

    filenames = os.listdir(dir)

    mi_list = [MedicalImage.load(os.path.join(dir, file))
               for file in filenames]

    q = 99.9
    cutoff = [60, 35]
    region_threshold = 0

    images = np.array([mi.images['pet'][cutoff[0]:-cutoff[1]]
                       for mi in mi_list])
    masks = [get_mask(img, q=q) for img in images]
    masks = mask_denoise(masks, min_region_size=region_threshold)

    # Normalization
    images = (2 * (images - np.min(images)) /
              (np.max(images) - np.min(images)) - 1)

    features = []
    for i, mi in enumerate(mi_list):
      mi.images['pet'] = images[i]
      mi.labels['label-0'] = masks[i]
      mi.crop([64, 256, 256], random_crop=False)
      features.append(mi.images['pet'])

    features = np.expand_dims(np.array(features), axis=-1)
    image_dict['features'] = features
    image_dict['targets'] = features

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts


def normalization_min_max(image: np.ndarray, range: list):
  assert len(range) == 2
  mult, add = range[1] - range[0], range[0]
  image = (mult * (image - np.min(image)) /
           (np.max(image) - np.min(image)) + add)

  return image


def get_mask(image, q: float):
  threshold = np.percentile(image, q=q)
  mask = image > threshold

  return mask.astype(np.uint8)


def mask_denoise(masks: list, min_region_size=50):
  '''

  '''
  from scipy.ndimage import label, generate_binary_structure

  labels = []
  structure = generate_binary_structure(3, 1)
  for mask in tqdm(masks):
    # Get connected region for denoise
    labeled_image, num_features = label(mask, structure)
    region_sizes = [np.sum(labeled_image == label)
                    for label in range(1, num_features + 1)]

    small_region_labels = [l for l, size in enumerate(region_sizes)
                           if size < min_region_size]

    for l in small_region_labels:
      labeled_image[labeled_image == l] = 0

    labeled_image[labeled_image > 1] = 1
    #
    indices = list(set(np.where(labeled_image == 1)[0]))
    bad_indices = find_noise_indices(indices)
    if bad_indices is not None:
      for i in bad_indices:
        labeled_image[i] = 0

    labels.append(labeled_image)

  return labels


def find_cutoff_values(lst: list, threshold):
  max_value = max(lst)
  max_index = lst.index(max_value)

  left_index, right_index = max_index, max_index

  while left_index > 0 and lst[left_index] >= threshold:
    left_index -= 1

  while right_index < len(lst) - 1 and lst[right_index] >= threshold:
    right_index += 1

  return left_index + 1, right_index - 1


def next_power_of_two(number):
  import math
  if number <= 0: return 1
  else:
    return 2 ** math.ceil(math.log2(number))


def find_noise_indices(indices: list):
  bad_indices = []
  total_list, current_sublist = [], []

  for num in indices:
    if not current_sublist or num - current_sublist[-1] == 1:
      current_sublist.append(num)
    else:
      total_list.append(current_sublist)
      current_sublist = [num]

  if current_sublist:
    total_list.append(current_sublist)

  if len(total_list) == 1: return []
  else:
    list_len = [len(l) for l in total_list]
    index = list_len.index(max(list_len))
    for l in range(len(total_list)):
      if l != index: bad_indices.extend(total_list[l])

  return bad_indices



if __name__ == '__main__':
  from bcp_core import th
  agent = BCPAgent()
  train_set, val_set, test_set = agent.load()
  print()





