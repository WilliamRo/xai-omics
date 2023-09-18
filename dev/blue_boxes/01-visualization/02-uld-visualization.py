import numpy as np

from dev.explorers.uld_explorer.uld_explorer_v3 import ULDExplorer
from xomics.data_io.uld_reader import UldReader
from xomics.data_io.utils.uld_raw_rd import rd_uld_test
from xomics import MedicalImage

data_dir = r'../../../data/01-ULD/'
subjects = [1, 6]
mode = "uld-outputs"


keys = [['Full'],
        # '1-2',
        # '1-4',
        # '1-10',
        # '1-20',
        ['1-50'],
        # '1-100',
        ]
mis = []
reader = UldReader(data_dir)

if mode == 'uld-train':
  mis = reader.load_data(subjects, keys, methods='mi')

if mode == 'uld-test':
  imgs = rd_uld_test(data_dir + "testset", subjects)
  for i in range(len(imgs)):
    mi = MedicalImage(f"test-{i}", {'low-test': imgs[i]})
    mis.append(mi)

if mode == 'uld-pair':
  data = reader.load_data(subjects, keys, methods='pair')
  data_f, data_t = {}, {}
  for i in range(len(subjects)):
    data_f[f'sub{i}'] = np.concatenate(data['features'])[i]
    data_t[f'sub{i}'] = np.concatenate(data['targets'])[i]
  mi_f = MedicalImage('features', data_f)
  mi_t = MedicalImage('targets', data_t)
  mis = [mi_f, mi_t]

if mode == 'uld-outputs':
  imgs_l = rd_uld_test(data_dir + "testset", subjects)
  imgs = rd_uld_test(data_dir + "testset", subjects, outputs=True)
  for i in range(len(imgs)):
    mi = MedicalImage(f"outputs-{i+1}", {'Full': imgs[i],
                                         'test-low': imgs_l[i]})
    mis.append(mi)

# Visualization
ue = ULDExplorer(mis)
ue.dv.set('vmin', auto_refresh=False)
ue.dv.set('vmax', auto_refresh=False)
ue.show()
