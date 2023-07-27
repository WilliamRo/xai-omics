from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



def load_npy_file(file_path):
  return np.load(file_path, allow_pickle=True).tolist()


input_dir = r'../../../data/00-CT-demo/'
ct_file = 'demo1_ct.npy'
label_file = 'demo1_label.npy'

# Loading data
# ct and label have the same shape
# [patient, slice, H, W]
ct = load_npy_file(os.path.join(input_dir, ct_file))
label = load_npy_file(os.path.join(input_dir, label_file))

# Normalization
ct = np.squeeze(ct) / np.max(ct)
ct_dose2 = ct + np.random.random(ct.shape) * 0.1

label = np.squeeze(label) / np.max(label)

mi_1 = MedicalImage('Patient-1', {'CT': ct, 'CT-dose2': ct_dose2},
                    {'Label-1': label})
mi_2 = MedicalImage('Patient-2', {'CT': ct[3:]}, {'Label-1': label[3:]})


# Visualization
dg = DrGordon([mi_1, mi_2])
dg.show()

