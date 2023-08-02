import os

from xomics.data_io.mi_reader import rd_data
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage
import numpy as np

data_dir = r'../../../data/01-ULD/'
subjects = ['Subject_1-6', 'Subject_7-12', 'Subject_13-18']
patient_num = 1


keys = ['Full_dose',
        # '1-2 dose',
        # '1-4 dose',
        '1-10 dose',
        # '1-20 dose',
        '1-50 dose',
        # '1-100 dose',
        ]
mis = []

# data.shape = [n_slice, H, w]
dose = {}
for dose_tag in keys:
  dose[dose_tag] = rd_data(data_dir, subjects, dose_tag, patient_num)
  print(dose[dose_tag].shape)

for i in range(len(subjects)*patient_num):
  img_dict = {}

  for key in dose.keys():
    img_dict[key] = dose[key][i]

  mi = MedicalImage(f'Patient-{i}', img_dict)
  mis.append(mi)

# Visualization
dg = DrGordon(mis)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
