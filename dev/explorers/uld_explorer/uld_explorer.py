from mpl_toolkits.axes_grid1 import make_axes_locatable

from xomics.data_io.uld_reader import UldReader
from xomics.gui.dr_gordon import DrGordon, SliceView, Plotter
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



class ULDExplorer(DrGordon):
  def __init__(self, medical_images, title='ULDose Explorer'):
    super().__init__(medical_images, title=title)

    self.dv = DoseViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.dv], overwrite=True)


class DoseViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)



class HistogramViewer(Plotter):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(self.view_hist, pictor=dr_gordon)

    self.new_settable_attr('xmin', None, float, 'Min x')
    self.new_settable_attr('xmax', None, float, 'Max x')
    self.new_settable_attr('ymin', None, float, 'Min y')
    self.new_settable_attr('ymax', None, float, 'Max y')


  def view_hist(self, ax: plt.Axes):
    mi: MedicalImage = self.pictor.get_element(self.pictor.Keys.PATIENTS)
    i_layer = self.pictor.get_element(self.pictor.Keys.LAYERS)
    key = list(mi.images.keys())[i_layer]
    x = np.ravel(mi.images[key])
    ax.hist(x=x, bins=50, log=True)

    ax.set_xlim(self.get('xmin'), self.get('xmax'))
    ax.set_ylim(self.get('ymin'), self.get('ymax'))



class DeltaViewer(SliceView):

  def __init__(self, dr_gordon=None, target_key='Full'):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)
    self.TARGET_KEY = target_key
    self.new_settable_attr('cmap', 'RdBu', str, 'Color map')
    self.new_settable_attr(
      'alpha', 1.0, float,
      'Coefficient before low dose image while calculating delta')

    self.settable_attributes.pop('vmin')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image

    # Show slice
    full_dose_slice = mi.images[self.TARGET_KEY][x]
    image: np.ndarray = mi.images[self.displayed_layer_key][x]
    delta = full_dose_slice - self.get('alpha') * image

    abs_vmax = (np.max(abs(delta)) if self.get('vmax') is None
                else self.get('vmax'))

    im = ax.imshow(delta, cmap=self.get('cmap'), vmin=-abs_vmax,
                   vmax=abs_vmax)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def optimize_alpha(self, stategy='alpha',
                     alphas: str = '0.9,0.95,1.0,1.05,1.1'):
    """Calculate losses given alphas"""
    assert stategy == 'alpha'
    mi: MedicalImage = self.selected_medical_image

    full = mi.images['Full']
    image: np.ndarray = mi.images[self.displayed_layer_key]

    alphas = [float(a) for a in alphas.split(',')]
    losses = [abs(np.mean(full - a * image)) for a in alphas]

    for l, a in zip(losses, alphas): print(f'.. loss({a}) = {l}')



if __name__ == '__main__':
  data_dir = r'../../../data/01-ULD/'

  subjects = [1]
  doses = [
    'Full',
    # '1-2',
    '1-4',
    # '1-10',
    '1-20',
    # '1-50',
    '1-100',
  ]
  reader = UldReader(data_dir)
  mi_list = reader.load_mi_data(subjects, doses, raw=True)

  ue = ULDExplorer(mi_list)
  ue.add_plotter(HistogramViewer())
  ue.dv.set('vmin', auto_refresh=False)
  ue.dv.set('vmax', auto_refresh=False)

  delta_viewer = DeltaViewer()
  delta_viewer.set('vmax', auto_refresh=False)
  ue.add_plotter(delta_viewer)

  ue.show()
