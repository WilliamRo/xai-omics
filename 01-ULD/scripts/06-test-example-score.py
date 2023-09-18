from utils.metrics_calc import get_metrics
from xomics.data_io.uld_reader import UldReader
from xomics.data_io.utils.raw_rw import rd_file

import numpy as np
import os




def load_seg(dirpath, fileid):
  filepath = os.path.join(dirpath, f'Anonymous_{fileid}.nii.gz')
  seg = rd_file(filepath)
  return seg


def process_with_seg(image, seg):
  return np.where(seg == 0, 0, image)


def calc_score(truth, predict, seg):
  global_metrics = ['SSIM', 'NRMSE', 'PSNR']
  global_scores = get_metrics(truth, predict, global_metrics)

  truth_seg = np.where(seg == 0, 0, truth)
  predict_seg = np.where(seg == 0, 0, predict)

  local_metrics = ['PSNR']
  local_scores = get_metrics(truth_seg, predict_seg, local_metrics)

  return global_scores, local_scores




if __name__ == '__main__':
  testpath = '../../data/01-ULD/testset/'
  seg_path = os.path.join(testpath, 'seg')
  reader = UldReader.load_as_npy_data(testpath, [1, 50],
                                      ('Anonymous_', '.nii.gz'))
  img = reader.data
  seg1 = load_seg(seg_path, 1)
  seg2 = load_seg(seg_path, 50)
  print(seg1.shape, seg2.shape)

  img1_g = img[0]
  img2_g = img[1]
  img1 = process_with_seg(img[0], seg1)
  img2 = process_with_seg(img[1], seg2)
  print(img1.shape, img2.shape)
