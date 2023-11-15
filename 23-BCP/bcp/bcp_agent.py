import copy
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
        parts[0], parts[1], parts[2],
        names=['TrainSet', 'ValSet', 'TestSet'], over_classes=True)
    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> BCPSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return BCPSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    # data_set = BCPSet(data_dict=image_dict, name='BCPSet')
    data_set = BCPSet(data_dict=image_dict, name='BCPSet', NUM_CLASSES=4,
                      classes=['left', 'right', 'both', 'normal'])

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
               for file in tqdm(filenames, desc='Reading mi files')]

    # there is a bad data named '118', 70/197
    mi_list = mi_list[:69] + mi_list[70:180] + mi_list[181:]

    # q = 99.85
    q = 99.9
    cutoff = [60, 35]
    region_threshold = 0
    slice_threshold = 100
    crop_size = [32, 128, 128]

    images = np.array([mi.images['pet'][cutoff[0]:-cutoff[1]]
                       for mi in mi_list])

    # Get as accurate a mask as possible
    masks = [get_mask(img, q=q) for img in images]
    masks = mask_denoise(masks, min_region_size=region_threshold,
                         slice_threshold=slice_threshold)

    # Normalization
    images = (images - np.min(images)) / (np.max(images) - np.min(images))

    # region: Test
    masks = amend_masks(masks)

    # endregion: Test

    features, labels = [], []
    for i, mi in tqdm(enumerate(mi_list), desc='storage mi'):
      mi.images['pet'] = images[i]
      mi.labels['label-0'] = masks[i]
      mi.crop(crop_size=crop_size, random_crop=False)
      features.append(mi.images['pet'] * mi.labels['label-0'])
      labels.append(mi.key)

    dg = DrGordon(mi_list)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.show()

    real_labels = np.array([l.split('-')[-1] for l in labels])
    e_t_n = {'left': 0, 'right': 1, 'both': 2, 'normal': 3}
    real_labels = np.vectorize(e_t_n.get)(real_labels)
    one_hot_labels = np.eye(4)[real_labels]

    features = np.expand_dims(np.array(features), axis=-1)
    image_dict['features'] = features
    image_dict['targets'] = one_hot_labels
    image_dict['labels'] = labels

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts


def get_mask(image, q: float):
  threshold = np.percentile(image, q=q)
  mask = image > threshold

  return mask.astype(np.uint8)


def mask_denoise(masks: list, min_region_size=50, slice_threshold=100):
  '''

  '''
  from scipy.ndimage import label, generate_binary_structure

  labels = []
  structure = generate_binary_structure(3, 1)
  for mask in tqdm(masks, desc='Mask denoising'):
    # Get connected region for denoise
    labeled_image, num_features = label(mask, structure)
    region_sizes = [np.sum(labeled_image == label)
                    for label in range(1, num_features + 1)]

    sorted_indices = sorted(enumerate(region_sizes), key=lambda x: x[1],
                            reverse=True)

    top_indices = [index for index, value in sorted_indices[:2]]

    region_num = [i for i in range(len(region_sizes))
                  if i not in top_indices]

    for l in region_num:
      labeled_image[labeled_image == l + 1] = 0

    # set the region with mask to 1
    labeled_image[labeled_image > 1] = 1

    # The slice with less than slice_threshold mask
    # in one slice is excluded
    slice_size = np.sum(labeled_image, axis=(1, 2))
    bad_indices = np.where(slice_size < slice_threshold)
    if bad_indices is not None:
      for i in bad_indices:
        labeled_image[i] = 0

    #
    indices = list(set(np.where(labeled_image == 1)[0]))
    bad_indices = find_noise_indices(indices)
    # assert bad_indices == []
    if bad_indices is not None:
      for i in bad_indices:
        labeled_image[i] = 0

    labels.append(labeled_image)

  return np.array(labels)


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


def amend_masks(masks: np.ndarray):
  new_masks = []
  for mask in tqdm(masks, desc='Mask amending'):
    indices = np.where(mask == 1)
    b_z, t_z = np.min(indices[0]), np.max(indices[0])
    b_x, t_x = np.min(indices[1]), np.max(indices[1])
    b_y, t_y = np.min(indices[2]), np.max(indices[2])
    mask[b_z:t_z+1, b_x:t_x+1, b_y:t_y+1] = 1
    new_masks.append(mask)

  return np.array(new_masks)


if __name__ == '__main__':
  from bcp_core import th
  agent = BCPAgent()
  train_set, val_set, test_set = agent.load()
  print()





