from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



input_dir = r'D:\XAI Dropbox\William Ro\02-William@ZJU\05-Collaborations\03-PET\2023-09-27\病灶分割\食管癌-mi\mi/'
input_dir = r'P:\xai-omics\data\04-Brain-CT-PET\mi'
mi_file = os.listdir(input_dir)[:10]

mi_list = []
for file in mi_file:
  mi_list.append(MedicalImage.load(os.path.join(input_dir, file)))

# Visualization
dg = DrGordon(mi_list)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()

