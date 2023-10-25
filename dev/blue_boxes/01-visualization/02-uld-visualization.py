import numpy as np

from dev.explorers.uld_explorer.uld_explorer_v31 import ULDExplorer
from xomics.data_io.uld_reader import UldReader
from xomics.data_io.scripts.uld_raw_rd import rd_uld_test
from xomics import MedicalImage

data_dir = r'../../../data/01-ULD/'
# subjects = list(range(1, 16))
subjects = [2]
mode = "uld-train"


keys = [['Full'],
        # '1-2',
        # '1-4',
        # '1-10',
        # ['1-20'],
        # ['1-50'],
        ['1-100'],
        ]
mis = []
reader = UldReader(data_dir)

if mode == 'uld-train':
  mis = reader.load_data(subjects, keys, methods='mi', use_suv=False,
                         raw=True)

if mode == 'uld-test':
  # imgs = rd_uld_test(data_dir + "testset", subjects)
  testpath = data_dir + "testset"
  reader = reader.load_as_npy_data(testpath, subjects,
                                   ('Anonymous_', '.nii.gz'), raw=True,
                                   shape=[600, 440, 440], use_suv=True
                                   )
  imgs = reader.data
  for i in range(len(imgs)):
    mi = MedicalImage(f"test-{i}", {'low-test': imgs[i]})
    mis.append(mi)

if mode == 'uld-pair':
  data = reader.load_data(subjects, keys, methods='pair', norm_margin=[0, 10, 0, 0, 0],
                          shape=[1, 684, 440, 440, 1])
  data_f, data_t = {}, {}
  for num, i in enumerate(subjects):
    data_f[f'sub{i}'] = np.concatenate(data['features'])[num]
    data_t[f'sub{i}'] = np.concatenate(data['targets'])[num]
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

if mode == 'uld-predict':
  import joblib
  filepath = ''
  mis = joblib.load(filepath)



# Visualization
ue = ULDExplorer(mis)
# ue = DrGordon(mis)
# ue.slice_view.set('vmax', auto_refresh=False)
ue.slice_view.set('vmin', auto_refresh=False)
ue.dv.set('vmin', auto_refresh=False)
ue.dv.set('vmax', auto_refresh=False)
ue.show()
