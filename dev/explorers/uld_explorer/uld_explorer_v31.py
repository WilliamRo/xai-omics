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
    self.new_settable_attr('delta_vmax', None, float,
                           'Max abs value for delta view')
    self.new_settable_attr('view_delta', False, bool, 'Option to view delta')

    self.new_settable_attr('dev_mode', False, bool, 'Developer mode')
    self.new_settable_attr('dev_arg', '1', str, 'Developer arguments')

    self.new_settable_attr('lock_full', True, bool,
                           'Option to lock full dose image')

    self.new_settable_attr('show_metric', False, bool, 'Option to show metric')
    self.new_settable_attr('show_slice_metric', True, bool,
                           'Option to show slice metric')
    self.new_settable_attr('axial_margin', 70, int,
                           'Margin to cut in axial view')
    self.new_settable_attr('show_weight_map', False, bool,
                           'Option to show weight map')

    self.new_settable_attr(
      'alpha', 1.0, float,
      'Coefficient before low dose image while calculating delta')

    self.new_settable_attr('slicomic', False, bool,
                           'Option to toggle slice-omics')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    full_dose_vol = mi.images['Full']
    selected_vol = mi.images[self.displayed_layer_key]

    # Show slice
    m = self.get('axial_margin')
    if m > 0:
      full_dose_slice = full_dose_vol[x][m:-m, m:-m]
      image: np.ndarray = selected_vol[x][m:-m, m:-m]
    else:
      full_dose_slice = full_dose_vol[x]
      image: np.ndarray = selected_vol[x]

    # Enhance image if required
    if self.get('dev_mode') and not all(
        [self.displayed_layer_key == 'Full' and self.get('lock_full')]):
      image = self.enhance(image)

    # Show weight map if required
    if self.get('show_weight_map') and mi.in_pocket('weight_map'):
      wm = mi.get_from_pocket('weight_map')[x]
      h, w, c = wm.shape
      # wm.shape = [H, W, C]
      if c < 3:
        zeros = np.zeros_like(image)
        # zeros = image / np.max(image)
        wm = np.concatenate([wm, zeros], axis=-1)
      else: wm = wm[:, :, :3]
      im = ax.imshow(wm)
    elif self.get('view_delta'):
      delta = full_dose_slice - self.get('alpha') * image
      abs_vmax = (np.max(abs(delta)) if self.get('delta_vmax') is None
                  else self.get('delta_vmax'))

      im = ax.imshow(delta, cmap=self.get('delta_cmap'), vmin=-abs_vmax,
                     vmax=abs_vmax)
    else:
      im = ax.imshow(image, cmap=self.get('cmap'), vmin=self.get('vmin'),
                     vmax=self.get('vmax'))

    # Set title
    title = ''
    if self.get('show_metric'):
      show_slice_metric = self.get('show_slice_metric')
      metrics = ['NRMSE', 'SSIM', 'PSNR']

      arr1, arr2 = full_dose_vol[..., 0], selected_vol[..., 0]
      if show_slice_metric:
        arr1, arr2 = full_dose_slice[..., 0], image[..., 0]

      def _get_title():
        if not show_slice_metric:
          print(f'>> Calculating metrics for `{self.displayed_layer_key}` ...')

        if self.displayed_layer_key == 'Full':
          return 'NRMSE: 0, SSIM: 1, PSNR: $\infty$'

        max_val = np.max(selected_vol)
        return ', '.join([f'{k}: {v:.4f}' for k, v in get_metrics(
          arr1, arr2, metrics, data_range=max_val).items()])

      # Find key
      alpha = np.sum(arr1) / np.sum(arr2)
      if show_slice_metric:
        title = _get_title()
      else:
        tt_key = self.displayed_layer_key + '_metrics'
        title = self.get_from_pocket(tt_key, initializer=_get_title)
        title = '[Global] ' + title

      title += r', $\alpha=$' + f'{alpha:.3f}'

      if self.get('dev_mode'): title = f'[Dev] {title}'

    if self.get('slicomic') and self.displayed_layer_key == 'Full':
      title += ', '.join(
        [f'{k}: {v:.4f}' for k, v in self.get_slice_omics(image).items()])

    # Show title if necessary
    if title != '': ax.set_title(title, fontsize=10)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  # region: Developer Area

  def profile_slice(self):
    mi = self.selected_medical_image

    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    if self.get('show_weight_map') and mi.in_pocket('weight_map'):
      wm = mi.get_from_pocket('weight_map')

      for c in range(wm.shape[-1]):
        h = np.max(wm[:, :, :, c], axis=(1, 2))
        ax.plot(h)

      ax.set_title('Weight Map Profile')
      ax.legend([f'Channel-{c}' for c in range(wm.shape[-1])])
    else:
      image = mi.images[self.displayed_layer_key]
      image = image[:, :, :, 0]

      h1 = np.max(image, axis=(1, 2))
      h2 = np.mean(image, axis=(1, 2))

      h1 = h1 / np.max(h1)
      h2 = h2 / np.max(h2)

      ax.plot(h1)
      ax.plot(h2)

      ax.set_title(f'{self.displayed_layer_key}')
      ax.legend(['Max', 'Avg'])

    fig.show()
  ps = profile_slice

  def show_slice_metrics(self):
    from roma import console

    mi: MedicalImage = self.selected_medical_image
    full = mi.images['Full']

    image_keys = [k for k in mi.images.keys() if k != 'Full']
    max_vals = {k: np.max(mi.images[k]) for k in image_keys}

    metric_keys = ['NRMSE', 'SSIM', 'PSNR']
    metrics = {k: {ik: [] for ik in image_keys} for k in metric_keys}
    for i in range(full.shape[0]):
      console.print_progress(i, full.shape[0])

      for ik in image_keys:
        md = get_metrics(full[i][..., 0], mi.images[ik][i][..., 0],
                         metric_keys, max_vals[ik])
        for k, v in md.items(): metrics[k][ik].append(v)

    console.show_status('Metrics calculated successfully!')

    fig = plt.figure()

    for i, k in enumerate(metric_keys):
      ax: plt.Axes = fig.add_subplot(len(metric_keys), 1, i + 1)
      for ik in image_keys:
        highlight = ik == 'Model-Output'
        ax.plot(metrics[k][ik],
                linewidth=2 if highlight else 1.2,
                alpha=1 if highlight else 0.7)
      ax.legend(image_keys)
      ax.set_ylabel(k)

    fig.show()
  ssm = show_slice_metrics

  def get_slice_omics(self, s: np.ndarray):
    features = {}
    features['max'] = np.max(s)
    features['mean'] = np.mean(s)
    return features

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

  # endregion: Developer Area

  # region: Shortcuts

  def register_shortcuts(self):
    super().register_shortcuts()
    self.register_a_shortcut(
      'space', lambda: self.flip('view_delta'), 'Toggle `view_delta`')
    self.register_a_shortcut(
      'm', lambda: self.flip('show_metric'), 'Turn on/off title')
    self.register_a_shortcut(
      'd', lambda: self.flip('dev_mode'), 'Toggle developer mode')
    self.register_a_shortcut(
      'L', lambda: self.flip('lock_full'), 'Toggle `lock_full`')
    self.register_a_shortcut(
      'S', lambda: self.flip('show_slice_metric'),
      'Turn on/off `show_slice_metric` option')
    self.register_a_shortcut(
      'o', lambda: self.flip('slicomic'), 'Toggle `slicomic`')
    self.register_a_shortcut(
      'w', lambda: self.flip('show_weight_map'), 'Toggle `show_weight_map`')

  # endregion: Shortcuts



if __name__ == '__main__':
  data_dir = r'../../../data/01-ULD/'
  # data_dir = r'F:\xai-omics-data\01-ULD'

  subjects = [2]
  doses = [
    'Full',
    # '1-2',
    # '1-4',
    # '1-10',
    # '1-20',
    '1-50',
    # '1-100',
  ]
  doses = [[i] for i in doses]
  reader = UldReader(data_dir)
  mi_list = reader.load_mi_data(subjects, doses, raw=True)

  ue = ULDExplorer(mi_list)
  ue.dv.set('vmin', auto_refresh=False)
  ue.dv.set('vmax', auto_refresh=False)

  ue.show()
