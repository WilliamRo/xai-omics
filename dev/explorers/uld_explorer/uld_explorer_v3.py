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

    self.new_settable_attr('delta_cmap', 'RdBu', str,
                           'Color map for delta view')
    self.new_settable_attr('delta_vmax', None, str,
                           'Max abs value for delta view')
    self.new_settable_attr('view_delta', False, bool, 'Option to view delta')

    self.new_settable_attr('dev_mode', False, bool, 'Developer mode')
    self.new_settable_attr('dev_arg', '5', str, 'Developer arguments')

    self.new_settable_attr('show_metric', False, bool, 'Option to show metric')
    self.new_settable_attr('show_slice_metric', True, bool,
                           'Option to show slice metric')

    self.new_settable_attr(
      'alpha', 1.0, float,
      'Coefficient before low dose image while calculating delta')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    full_dose_vol = mi.images['Full']
    selected_vol = mi.images[self.displayed_layer_key]

    # Show slice
    full_dose_slice = full_dose_vol[x]
    image: np.ndarray = selected_vol[x]

    # Enhance image if required
    if self.get('dev_mode'):
      image = self.enhance(image)

    if self.get('view_delta'):
      delta = full_dose_slice - self.get('alpha') * image
      abs_vmax = (np.max(abs(delta)) if self.get('delta_vmax') is None
                  else self.get('delta_vmax'))

      im = ax.imshow(delta, cmap=self.get('delta_cmap'), vmin=-abs_vmax,
                     vmax=abs_vmax)
    else:
      im = ax.imshow(image, cmap=self.get('cmap'), vmin=self.get('vmin'),
                     vmax=self.get('vmax'))

    # Set title
    if self.get('show_metric'):
      show_slice_metric = self.get('show_slice_metric')
      metrics = ['NRMSE', 'SSIM', 'PSNR']

      arr1, arr2 = full_dose_vol[..., 0], selected_vol[..., 0]
      if show_slice_metric:
        arr1, arr2 = full_dose_slice[..., 0], image[..., 0]

      def _get_title():
        if not show_slice_metric:
          print(f'>> Calculating metrics for `{self.displayed_layer_key}` ...')
        max_val = np.max(selected_vol)
        return ', '.join([f'{k}: {v:.4f}' for k, v in get_metrics(
          arr1, arr2, metrics, data_range=max_val).items()])

      # Find key
      if show_slice_metric:
        title = _get_title()
      else:
        tt_key = self.displayed_layer_key + '_metrics'
        title = self.get_from_pocket(tt_key, initializer=_get_title)
        title = '[Global] ' + title

      ax.set_title(title)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def enhance(self, x: np.ndarray):
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(x, sigma=float(self.get('dev_arg')))
    return x


  def set_dev_arg(self, v):
    """Set developer argument"""
    self.set('dev_arg', v)
  da = set_dev_arg


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
      'space', lambda: self.flip('view_delta'), 'Toggle `view_delta`')
    self.register_a_shortcut(
      'm', lambda: self.flip('show_metric'), 'Turn on/off title')
    self.register_a_shortcut(
      'd', lambda: self.flip('dev_mode'), 'Toggle developer mode')
    self.register_a_shortcut(
      'S', lambda: self.flip('show_slice_metric'),
      'Turn on/off `show_slice_metric` option')



if __name__ == '__main__':
  data_dir = r'../../../data/01-ULD/'

  subjects = [1]
  doses = [
    'Full',
    # '1-2',
    # '1-4',
    # '1-10',
    '1-20',
    # '1-50',
    # '1-100',
  ]
  doses = [[i] for i in doses]
  reader = UldReader(data_dir)
  mi_list = reader.load_data(subjects, doses, methods='mi', raw=True)

  ue = ULDExplorer(mi_list)
  ue.dv.set('vmin', auto_refresh=False)
  ue.dv.set('vmax', auto_refresh=False)

  ue.show()
