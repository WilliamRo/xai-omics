from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics_calc import get_metrics

from xomics.data_io.uld_reader import UldReader
from xomics.gui.dr_gordon import DrGordon, SliceView, Plotter
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



class BrainExplorer(DrGordon):

  def __init__(self, medical_images, title='Brain Explorer'):
    super().__init__(medical_images, title=title)
    self.brain_viewer = BrainViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.brain_viewer], overwrite=True)


  def _refresh(self, wait_for_idle=False):
    if not wait_for_idle and not self.get_from_pocket(
        self.Keys.ALLOW_MAIN_THREAD_REFRESH, default=True):
      return
    # Refresh title
    self.title = f'{self.cursor_string} {self.static_title}{self.title_suffix}'
    # Refresh canvas
    self.canvas_refresh(wait_for_idle)


  def canvas_refresh(self, wait_for_idle=False):
    canvas = self.canvas
    # Clear figure
    # canvas._clear()
    if canvas.in_pocket(canvas.Keys.AXES2D):
      ax: plt.Axes = canvas.get_from_pocket(canvas.Keys.AXES2D)
      ax.clear()

    # Call active plotter
    canvas.pictor.active_plotter()

    # Tight layout and refresh
    # canvas.figure.tight_layout()

    if wait_for_idle: canvas._canvas.draw_idle()
    else: canvas._canvas.draw()



class BrainViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)

  def register_shortcuts(self):
    super().register_shortcuts()

  def mask(self, q: float):
    x = self.selected_medical_image.images['pet']
    threshold = np.percentile(x, q=q)
    self.selected_medical_image.labels['mask'] = x > threshold

  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image

    # Show slice
    image: np.ndarray = mi.images[self.displayed_layer_key][x]
    im = ax.imshow(image, cmap=self.get('cmap'), vmin=self.get('vmin'),
                   vmax=self.get('vmax'))

    ax.set_title(f'{self.selected_medical_image.key}')

    # Show annotations
    for i, anno_key in enumerate(self.annotations_to_show):
      mask = mi.labels[anno_key][x]
      anno = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
      anno[..., i % 3] = 1.0
      anno[..., 3] = mask
      ax.imshow(anno, alpha=0.2)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)



if __name__ == '__main__':
  data_dir = r'../../../data/04-Brain-CT-PET/mi'

  subjects = [3, 4, 5]

  mi_list = []
  for file in [os.path.join(data_dir, f'{s}.mi') for s in subjects]:
    mi_list.append(MedicalImage.load(file))

  be = BrainExplorer(mi_list)
  be.brain_viewer.set('vmin', auto_refresh=False)
  be.brain_viewer.set('vmax', auto_refresh=False)

  be.show()
