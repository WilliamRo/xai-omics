# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from utils.metrics_calc import get_metrics
#
# from xomics.data_io.uld_reder import UldReader
from xomics.gui.dr_gordon import DrGordon, SliceView, Plotter
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



class BrainExplorer(DrGordon):

  def __init__(self, medical_images, title='Brain Explorer'):
    super().__init__(medical_images, title=title)
    self.brain_viewer = BrainViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.brain_viewer],
                     overwrite=True)



class BrainViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)

  def register_shortcuts(self):
    super().register_shortcuts()

  def mask(self, q: float):
    """Generate mask"""
    # (0) Get 3D brain volume
    x = list(self.selected_medical_image.images.values())[0]
    # x.shape = [S, H, W, 1]

    # (1) Create mask based on x
    threshold = np.percentile(x, q=q)
    mask = x > threshold

    # (*) Visualize your mask
    self.selected_medical_image.labels['mask-1'] = mask

    cn_mask = x > np.percentile(x, 99)
    self.selected_medical_image.labels['caudate nuclei'] = cn_mask



if __name__ == '__main__':
  data_dir = r'../../../data/05-Brain-MR/mi/neurocon/'

  subjects = ['sub-control032014', 'sub-control032015']

  mi_list = []
  for file in [os.path.join(data_dir, f'{s}.mi') for s in subjects]:
    mi_list.append(MedicalImage.load(file))

  be = BrainExplorer(mi_list)
  be.brain_viewer.set('vmin', auto_refresh=False)
  be.brain_viewer.set('vmax', auto_refresh=False)

  be.show()
