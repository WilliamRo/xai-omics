import os

from xomics.data_io.mi_reader import load_data
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage
import numpy as np

data_dir = r'../../../data/01-ULD/'
subjects = [1, 3, 5, 10, 12]
patient_num = 1


keys = ['Full',
        # '1-2',
        '1-4',
        # '1-10',
        # '1-20',
        # '1-50',
        # '1-100',
        ]
mis = []

for subject in subjects:
  img = load_data(data_dir, subject, keys)
  data_dict = dict(zip(keys, img))
  mi = MedicalImage(f'Subject-{subject}', data_dict)
  mis.append(mi)

# Visualization
dg = DrGordon(mis)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
