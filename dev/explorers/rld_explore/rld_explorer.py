from mpl_toolkits.axes_grid1 import make_axes_locatable

from xomics.data_io.utils.metrics_calc import get_metrics
from xomics.gui.dr_gordon import DrGordon, SliceView
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np



class RLDExplorer(DrGordon):

  def __init__(self, medical_images, title='RLD Explorer'):
    super().__init__(medical_images, title=title)

    self.sv = RLDViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.sv], overwrite=True)

    self.shortcuts.register_key_event(
      ['l'], lambda: self.set_cursor(self.Keys.LAYERS, 1, refresh=True),
      description='Next layer', color='yellow')
    self.shortcuts.register_key_event(
      ['h'], lambda: self.set_cursor(self.Keys.LAYERS, -1, refresh=True),
      description='Previous layer', color='yellow')


class RLDViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)

    self.new_settable_attr('view_point', 'other', str,
                           'Change the methods to view(cor, sag, other)')
    self.new_settable_attr('show_metric', False, bool, 'Option to show metric')
    self.new_settable_attr('show_slice_metric', True, bool,
                           'Option to show slice metric')
    self.new_settable_attr('full_key', '', str,
                           'the key of the full image')
    self.new_settable_attr('show_weight_map', False, bool,
                           'Option to show weight map')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    selected_vol = mi.images[self.displayed_layer_key][::-1]
    if self.get('full_key') == '':
      self.set('full_key', list(mi.images.keys())[-2])
    full_key = self.get('full_key')
    full_dose_vol: np.ndarray = mi.images[full_key][::-1]

    ps = [0.9765625]*2
    ss = 3.0
    aspect = ps[1]/ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    if self.get('view_point') == 'cor':
      selected_vol = selected_vol.swapaxes(0, 1)
      aspect = cor_aspect
    elif self.get('view_point') == 'sag':
      selected_vol = selected_vol.swapaxes(0, 2)
      aspect = sag_aspect

    image: np.ndarray = selected_vol[x]
    full_dose_slice = full_dose_vol[x]
    if self.get('show_weight_map') and mi.in_pocket('weight_map'):
      wm = mi.get_from_pocket('weight_map')[::-1][x]
      h, w, c = wm.shape
      # wm.shape = [H, W, C]
      if c < 3:
        zeros = np.zeros_like(image)
        # zeros = image / np.max(image)
        wm = np.concatenate([wm, zeros], axis=-1)
      else: wm = wm[:, :, :3]
      im = ax.imshow(wm)
    else:
      im = ax.imshow(image, cmap=self.get('cmap'), vmin=self.get('vmin'),
                     vmax=self.get('vmax'))
    ax.set_aspect(aspect)

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

        if self.displayed_layer_key == full_key:
          return 'NRMSE: 0, SSIM: 1, PSNR: $\infty$'

        max_val = np.max(selected_vol)
        return ', '.join([f'{k}: {v:.4f}' for k, v in get_metrics(
          arr1, arr2, metrics, data_range=max_val).items()])

      if show_slice_metric:
        title = _get_title()
      else:
        tt_key = self.displayed_layer_key + '_metrics'
        title = self.get_from_pocket(tt_key, initializer=_get_title)
        title = '[Global]' + title

    # Show title if necessary
    ax.set_title(f'[{mi.key}]{title}', fontsize=10)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

  # region: Custom

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
        highlight = ik == 'Output'
        ax.plot(metrics[k][ik],
                linewidth=2 if highlight else 1.2,
                alpha=1 if highlight else 0.7)
      ax.legend(image_keys)
      ax.set_ylabel(k)

    fig.show()
  ssm = show_slice_metrics

  def show_slice_3d(self, i: int, j: int):
    mi: MedicalImage = self.selected_medical_image
    selected_vol = mi.images[self.displayed_layer_key][::-1]
    selected_slice: np.ndarray = selected_vol[i]

    pixel_values = selected_slice[:, ..., 0][j]
    # print(selected_slice.shape)
    x = np.arange(selected_slice.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, pixel_values)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Visualization-{i}-{j}')
    # ax.view_init(elevation_angle, azimuthal_angle)
    plt.show()
  ss3 = show_slice_3d

  # region: Shortcuts

  def register_shortcuts(self):
    super().register_shortcuts()

    def modify_view():
      vlist = ['other', 'cor', 'sag']
      self.set('view_point', vlist[(vlist.index(self.get('view_point'))+1) % 3])
      self.refresh()

    self.register_a_shortcut(
      'v', modify_view, 'Change the Viewpoints')
    self.register_a_shortcut(
      'm', lambda: self.flip('show_metric'), 'Turn on/off title')
    self.register_a_shortcut(
      'S', lambda: self.flip('show_slice_metric'),
      'Turn on/off `show_slice_metric` option')
    self.register_a_shortcut(
      'w', lambda: self.flip('show_weight_map'), 'Toggle `show_weight_map`')

  # endregion: Shortcuts



if __name__ == '__main__':
  pass
