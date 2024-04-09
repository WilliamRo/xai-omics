import numpy as np

from xomics import MedicalImage
from xomics.objects.jutils.general_mi import GeneralMI

if __name__ == '__main__':
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer
  csv_path = r'../../data/02-RLD/rld_data.csv'

  test = GeneralMI.get_test_sample(csv_path)
  test.process_param['norm'] = 'PET'
  test.process_param['shape'] = [440, 440, 560]
  # test.process_param['percent'] = 99.9
  # test.process_param['ct_window'] = [50, 500]

  # test.LOW_MEM = True
  num = 0
  test = test[1:2]
  img1 = test.images['240S'][num][200:201]
  img2 = test.images['30G'][num][200:201]
  sino = test.radon_transform(img1[0])
  raw = test.radon_reverse(sino)
  sino = np.expand_dims(sino, axis=0)
  noise = sino[0] + np.random.poisson(sino[0])
  noise = test.radon_reverse(noise)
  noise = np.expand_dims(noise, axis=0)
  raw = np.expand_dims(raw, axis=0)

  # onehot = test.mask2onehot(test.images['CT_seg'][0], [5, 10, 11, 12, 13, 14, 51])
  print(img1.shape, img2.shape)

  mi = MedicalImage(test.pid[num], images={'t1': img1, 'noise': noise,
                                           'test': sino, 'low': img2})
  re = RLDExplorer([mi])
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.sv.set('cmap', 'gist_yarg')
  re.show()