from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage
from tqdm import tqdm

import numpy as np
import os



# mi_dir = r'../../data/BA & MIA/mi/BA'
mi_dir = r'E:\PET\Brain Seg\data\mi'
mi_file_list = os.listdir(mi_dir)
mi_list = []

indice = []
for file in tqdm(mi_file_list):
  mi: MedicalImage = MedicalImage.load(os.path.join(mi_dir, file))
  mi_list.append(mi)
  if len(mi_list) >= 5: break


# Visualization
dg = DrGordon(mi_list)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
