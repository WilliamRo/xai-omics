from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics_calc import get_metrics

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

    self.shortcuts.register_key_event(
      ['l'], lambda: self.set_cursor(self.Keys.LAYERS, 1, refresh=True),
      description='Next layer', color='yellow')
    self.shortcuts.register_key_event(
      ['h'], lambda: self.set_cursor(self.Keys.LAYERS, -1, refresh=True),
      description='Previous layer', color='yellow')

  # region: Data IO

  @staticmethod
  def load_subject(data_dir, subject, doses):
    data_dict = {}
    for dose in doses:
      file_path = os.path.join(data_dir, f'subject{subject}',
                              f'subject{subject}_{dose}.npy')
      data_dict[f'{dose}'] = np.load(file_path)[0]
    return MedicalImage(f'Subject-{subject}', data_dict)

  # endregion: Data IO



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
    self.new_settable_attr('show_metric', False, bool, 'Option to show metric')
    self.new_settable_attr(
      'alpha', 1.0, float,
      'Coefficient before low dose image while calculating delta')

    self.settable_attributes.pop('vmin')

    self.set('color_bar', True, auto_refresh=False)


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    full_dose_vol = mi.images[self.TARGET_KEY]
    selected_vol = mi.images[self.displayed_layer_key]

    # Show slice
    full_dose_slice = full_dose_vol[x]
    image: np.ndarray = selected_vol[x]

    delta = full_dose_slice - self.get('alpha') * image

    abs_vmax = (np.max(abs(delta)) if self.get('vmax') is None
                else self.get('vmax'))

    im = ax.imshow(delta, cmap=self.get('cmap'), vmin=-abs_vmax,
                   vmax=abs_vmax)

    # Set title
    if self.get('show_metric'):
      metrics = ['NRMSE', 'SSIM', 'PSNR']

      def _get_title():
        print(f'>> Calculating metrics for `{self.displayed_layer_key}` ...')
        max_val = np.max(selected_vol)
        return ', '.join([f'{k}: {v:.4f}' for k, v in get_metrics(
          np.reshape(full_dose_vol, newshape=full_dose_vol.shape[:3]),
          np.reshape(selected_vol, newshape=selected_vol.shape[:3]),
          metrics, data_range=max_val).items()])
      tt_key = self.displayed_layer_key + '_metrics'
      title = self.get_from_pocket(tt_key, initializer=_get_title)

      ax.set_title(title)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def optimize_alpha(self):
    """Calculate losses given alphas"""
    mi: MedicalImage = self.selected_medical_image

    full = mi.images['Full']
    image: np.ndarray = mi.images[self.displayed_layer_key]

    self.set('alpha', np.sum(full) / np.sum(image))
  oa = optimize_alpha


  def register_shortcuts(self):
    super().register_shortcuts()
    self.register_a_shortcut(
      'M', lambda: self.flip('show_metric'), 'Turn on/off title')



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
  doses = [[i] for i in doses]
  reader = UldReader(data_dir)
  mi_list = reader.load_data(subjects, doses, methods='mi', raw=True)

  ue = ULDExplorer(mi_list)
  # ue.add_plotter(HistogramViewer())
  ue.dv.set('vmin', auto_refresh=False)
  ue.dv.set('vmax', auto_refresh=False)

  delta_viewer = DeltaViewer()
  delta_viewer.set('vmax', auto_refresh=False)
  ue.add_plotter(delta_viewer)

  ue.show()
