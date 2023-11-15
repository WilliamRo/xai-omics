from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage
from tqdm import tqdm

import numpy as np
import os



# mi_dir = r'../../data/BA & MIA/mi/BA'
mi_dir = r'E:\PET\Brain Seg\data\mi'
raw_dir = r'../../data/02-PET-CT-Y1/results/04-prt/mi'
model_name = '1111_prt(8-5-2-1-relu-mp)_Sc_23'
# model_name = '1111_prt(8-5-2-3-relu-mp)_Sc_26'
# model_name = '1112_prt(8-5-2-3-relu-mp)_Sc_29'
# model_name = '1113_prt(8-5-2-3-relu-mp)_Sc_31'
# model_name = '1114_prt(4-5-2-3-relu-mp)_Sc_3'

mi_dir = os.path.join(raw_dir, model_name)
mi_file_list = os.listdir(mi_dir)
mi_list = []

indice = []
for file in tqdm(mi_file_list):
  mi: MedicalImage = MedicalImage.load(os.path.join(mi_dir, file))
  mi_list.append(mi)


# Visualization
dg = DrGordon(mi_list)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
