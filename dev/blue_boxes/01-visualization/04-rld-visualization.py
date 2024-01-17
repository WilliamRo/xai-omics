import numpy as np
import os

from xomics import MedicalImage
from xomics.objects.general_mi import GeneralMI
from dev.explorers.rld_explore.rld_explorer import RLDExplorer

data_dir = r'../../../data/02-RLD'
subject = [8, 9]

data = np.genfromtxt(os.path.join(data_dir, 'rld_data.csv'), delimiter=',', dtype=str)
types = data[0][1:]
pid = data[1:, 0]
path_array = data[1:, 1:]

img_dict = {}
for i, type_name in enumerate(types):
  img_path = path_array[:, i]
  img_dict[type_name] = {'path': img_path}

process_param = {
  'ct_window': None,
  'norm': None,  # only min-max,
  'shape': [440, 440, 263],  # [320, 320, 240]
  'crop': [10, 0, 0][::-1],  # [30, 30, 10]
  'clip': None,  # [1, None]
}

img_type = {
  'CT': ['CT'],
  'PET': ['30G', '20S', '40S', '60G', '120S', '240G', '240S'],
  'MASK': ['CT_seg'],
  'STD': ['30G'],
}

mi = GeneralMI(img_dict, process_param=process_param, pid=pid, img_type=img_type)
mi.image_keys = ['30G', '240G', 'CT']
subs = [0]
print(mi.images[0][0].shape, mi.images[2][0].shape)
mis = [
  MedicalImage(f'{mi.pid[i]}', images={
    mi.image_keys[j]: mi.images[j][i] for j in range(len(mi.image_keys))
  }) for i in subs
]

# mis = mi[[0, 1]]
# for i in range(2):
#   print(mis.images[i][0])
re = RLDExplorer(mis)
re.sv.set('vmin', auto_refresh=False)
re.sv.set('vmax', auto_refresh=False)
re.sv.set('cmap', 'gist_yarg')
re.show()
