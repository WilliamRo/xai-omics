import copy
import random
import os
import numpy as np
import SimpleITK as sitk

from bcp.bcp_set import BCPSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from copy import deepcopy
from xomics.gui.dr_gordon import DrGordon
from scipy.ndimage import label, generate_binary_structure


class BCPAgent(DataAgent):

  TFD_FILE_NAME = 'bcp.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from bcp_core import th
    th.ratio_of_dataset = '6:1:3'

    bcp_set = cls.load_as_tframe_data(th.data_dir)
    mi_list = bcp_set.data_dict['mi_list'].tolist()

    # Pre-processing
    for mi in tqdm(mi_list, desc='Pre processing'):
      assert isinstance(mi, MedicalImage)
      mi.normalization(['mr'], 'min_max')
      mi.images['mr'] = mi.images['mr'] * 2 - 1

      bottom, top = mi.crop([64, 64, 64], False, ['caudate'])
      mi.put_into_pocket('crop_info', [bottom, top])

    if len(bcp_set) == 1:
      # If there is only 1 example, we make training set, validation set
      # and testing set the same.
      datasets = bcp_set, bcp_set, bcp_set
    else:
      ratio = [int(a) for a in th.ratio_of_dataset.split(':')]
      part = ratio_to_realnum(ratio, len(mi_list))

      train_list = mi_list[:part[0]]
      val_list = mi_list[part[0]:part[0] + part[1]]
      test_list = mi_list[-part[2]:]

      ds_list = [train_list, val_list, test_list]
      ds_name = ['TrainSet', 'ValidSet', 'TestSet']
      datasets = [
        BCPSet(data_dict={'mi_list': np.array(dl, dtype=object)}, name=dn)
        for dl, dn in zip(ds_list, ds_name)]

    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> BCPSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return BCPSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    # data_set = BCPSet(data_dict=image_dict, name='BCPSet')
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

    brain_dir = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/05-Brain-MR/yaojie_mr_brain')
    pids = os.listdir(brain_dir)
    mi_list = []
    for p in tqdm(pids, desc='Loading data'):
      mr_path = os.path.join(brain_dir, p, 'fastsurfer_mr.nii')
      mask_path = os.path.join(brain_dir, p, f'hand-{p.split("-")[-1]}-caudate.nii')

      mr_image = sitk.ReadImage(mr_path)
      mask_image = sitk.ReadImage(mask_path)

      mr_array = sitk.GetArrayFromImage(mr_image)
      mask_array = sitk.GetArrayFromImage(mask_image)

      assert mr_array.shape == mask_array.shape

      mi = MedicalImage(
        images={'mr': mr_array}, labels={'caudate': mask_array}, key=p)

      mi.put_into_pocket(mi.Keys.SPACING, mr_image.GetSpacing(), local=True)
      mi.put_into_pocket(mi.Keys.DIRECTION, mr_image.GetDirection(), local=True)
      mi_list.append(mi)

    image_dict['mi_list'] = np.array(mi_list, dtype=object)

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts



if __name__ == '__main__':
  from bcp_core import th
  agent = BCPAgent()
  train_set, val_set, test_set = agent.load()
  print()





