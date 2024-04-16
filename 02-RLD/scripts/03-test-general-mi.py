import numpy as np

from time import time
from xomics import MedicalImage
from xomics.objects.jutils.objects import GeneralMI

if __name__ == '__main__':
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer
  csv_path = r'../../data/02-RLD/rld_data.csv'

  test = GeneralMI.get_test_sample(csv_path)
  # test.process_param['percent'] = 99.9
  # test.process_param['ct_window'] = [50, 500]

  # test.LOW_MEM = True
  # num = 0
  # test = test[1:2]
  # img1 = test.images['240S'][num][200:201]
  # img2 = test.images['30G'][num][200:201]
  #
  # sino = test.radon_transform(img1[0])
  # sino = np.expand_dims(sino, axis=0)
  #
  # noise = sino[0] + np.random.poisson(img1[0])
  # noise = test.radon_reverse(noise)
  # noise = np.expand_dims(noise, axis=0)


  # onehot = test.mask2onehot(test.images['CT_seg'][0], [5, 10, 11, 12, 13, 14, 51])
  # print(img1.shape, img2.shape)
  #
  # mi = MedicalImage(test.pid[num], images={'t1': img1, 'noise': noise,
  #
  mis = []
  index = [1,2]

  imgs30 = test.itk_imgs['30G'][index]
  imgs240 = test.images['240S'][index]


  for img30, img240, i in zip(imgs30, imgs240, index):
    mi = MedicalImage(test.pid[i], images={'test': img30, 'full': img240})
    mis.append(mi)

  re = RLDExplorer(mis)
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.sv.set('cmap', 'gist_yarg')
  re.sv.set('full_key', 'full')
  re.show()