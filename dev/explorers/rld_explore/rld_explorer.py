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

  def refresh(self, wait_for_idle=False):
    self.title_suffix = f' - [{self.sv.selected_medical_image.key}] ' \
                        f'{self.sv.displayed_layer_key}'
    super(DrGordon, self).refresh(wait_for_idle)


class RLDViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)
    self.out_fig, self.out_ax = plt.subplots()
    self.pro_method = 'h'
    self.pro_cursor = 220

    self.new_settable_attr('view_point', 'other', str,
                           'Change the methods to view(cor, sag, other)')
    self.new_settable_attr('show_metric', False, bool, 'Option to show metric')
    self.new_settable_attr('show_slice_metric', True, bool,
                           'Option to show slice metric')
    self.new_settable_attr('full_key', '', str,
                           'the key of the full image')
    self.new_settable_attr('show_weight_map', False, bool,
                           'Option to show weight map')
    self.new_settable_attr('show_suv_metric', False, bool,
                           'Option to show suv metric')
    self.new_settable_attr('show_profile', False, bool,
                           'Option to show profile')
    self.new_settable_attr('view_delta', False, bool, 'Option to view delta')
    self.new_settable_attr('delta_cmap', 'RdBu', str,
                           'Color map for delta view')
    self.new_settable_attr('delta_vmax', None, float,
                           'Max abs value for delta view')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    # fig.suptitle(f'[{mi.key}]' + self.displayed_layer_key)
    selected_vol = mi.images[self.displayed_layer_key][::-1]
    if self.get('full_key') == '':
      self.set('full_key', list(mi.images.keys())[-2])
    full_key = self.get('full_key')
    full_dose_vol: np.ndarray = mi.images[full_key][::-1]

    ps = [1.65]*2
    ss = 3.0
    aspect = ps[1]/ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    if self.get('view_point') == 'cor':
      selected_vol = selected_vol.swapaxes(0, 1)
      full_dose_vol = full_dose_vol.swapaxes(0, 1)
      aspect = cor_aspect
    elif self.get('view_point') == 'sag':
      selected_vol = selected_vol.swapaxes(0, 2)
      full_dose_vol = full_dose_vol.swapaxes(0, 2)
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
    elif self.get('view_delta'):
      delta = full_dose_slice - image
      abs_vmax = (np.max(abs(delta)) if self.get('delta_vmax') is None
                  else self.get('delta_vmax'))

      im = ax.imshow(delta, cmap=self.get('delta_cmap'), vmin=-abs_vmax,
                     vmax=abs_vmax)
    else:
      simage = image.copy()

      if self.get("show_profile"):
        if self.get('full_key') == self.displayed_layer_key:
          self.show_profile([mi.images[k][::-1][x] for k in mi.images.keys()],
                            list(mi.images.keys()))
        else:
          self.show_profile([image, full_dose_slice],
                            [self.displayed_layer_key, 'Full'])
        if self.pro_method == 'h':
          simage[self.pro_cursor] = np.max(image)
        else:
          simage[:, self.pro_cursor] = np.max(image)

      im = ax.imshow(simage, cmap=self.get('cmap'), vmin=self.get('vmin'),
                     vmax=self.get('vmax'))
    ax.set_aspect(aspect)

    # Set title
    title = ''
    if self.get('show_metric'):
      show_slice_metric = self.get('show_slice_metric')
      metrics = ['NRMSE', 'SSIM', 'PSNR', 'RELA']

      arr1, arr2 = full_dose_vol, selected_vol
      if show_slice_metric:
        arr1, arr2 = full_dose_slice, image

      def _get_title():
        if not show_slice_metric:
          print(f'>> Calculating metrics for `[{self.selected_medical_image.key}]'
                f'{self.displayed_layer_key}` ...')

        if self.displayed_layer_key == full_key:
          return 'NRMSE: 0, SSIM: 1, PSNR: $\infty$'

        max_val = np.max(full_dose_vol)
        return ', '.join([f'{k}: {v:.4f}' for k, v in get_metrics(
          arr1 / max_val, arr2 / max_val, metrics, data_range=1.0).items()])

      if show_slice_metric:
        title = _get_title()
      else:
        tt_key = self.displayed_layer_key + f'_{self.selected_medical_image.key}' \
                                            f'_metrics'
        title = self.get_from_pocket(tt_key, initializer=_get_title)
        title = '[Global]' + title
    elif self.get('show_suv_metric'):
      show_slice_metric = self.get('show_slice_metric')
      if show_slice_metric:
        mimage = image
        # fimage = full_dose_slice
      else:
        mimage = selected_vol
        # fimage = full_dose_vol
        title = '[Global]'
      suv_mean = np.mean(mimage)
      suv_max = np.max(mimage)
      suv_median = np.median(mimage)
      title += r"$SUV_{max}=$%.4f, $SUV_{mean}=$%.4f, $SUV_{median}=$%.4f" % \
               (suv_max, suv_mean, suv_median)

    # Show title if necessary
    ax.set_title(f'{title}', fontsize=10)

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

  def show_profile(self, image_list, legend):
    fig: plt.figure = self.out_fig
    ax = self.out_ax
    ax.clear()
    shape = image_list[0].shape[1]
    x = np.arange(shape)

    for image in image_list:
      if self.pro_method == 'h':
        values = image[self.pro_cursor]
      else:
        values = image[:, self.pro_cursor].reshape(shape)
      ax.plot(x, values)

    ax.legend(legend)
    ax.set_ylabel('SUV')

    ax.set_xlabel('Width')
    ax.set_title(f'Profile {self.pro_method.upper()}:{self.pro_cursor}')
    # ax.view_init(elevation_angle, azimuthal_angle)
    fig.canvas.draw()

  def union_joint_hist(self):
    mi: MedicalImage = self.selected_medical_image
    image_keys = [k for k in mi.images.keys() if k != 'Full']
    full = mi.images['Full'].ravel()
    print(image_keys, mi.images['Full'].shape)
    fig = plt.figure()
    min_val = 1
    max_val = 5

    for i, key in enumerate(image_keys):
      ax: plt.Axes = fig.add_subplot(1, 2, i+1)
      img = mi.images[key].ravel()
      hist2D, x_edges, y_edges = np.histogram2d(img, full, bins=2048,
                                                range=[[min_val, max_val],
                                                       [min_val, max_val]])

      im = ax.imshow(hist2D.T, origin='lower', cmap='hsv',
                     extent=[min_val, max_val, min_val, max_val])
      ax.set_xticks(np.arange(min_val, max_val + 1))
      ax.set_yticks(np.arange(min_val, max_val + 1))
      ax.set_xlabel(f'{key} PET')
      ax.set_ylabel('Full Dose PET')
      ax.set_title('Joint Voxel Histogram')
      fig.colorbar(im)

    fig.show()

  ujh = union_joint_hist
  # region: Shortcuts

  def register_shortcuts(self):
    super().register_shortcuts()

    def modify_view():
      vlist = ['other', 'cor', 'sag']
      self.set('view_point', vlist[(vlist.index(self.get('view_point'))+1) % 3])
      self.refresh()

    def pro_move(i):
      self.pro_cursor += i
      if self.pro_cursor < 0: self.pro_cursor = 0
      if self.pro_cursor >= 440: self.pro_cursor = 439
      self.refresh()

    def show_profile():
      self.flip('show_profile')
      if self.get('show_profile'):
        self.out_fig.show()

    def switch_pro_method():
      if self.pro_method == 'h':
        self.pro_method = 'w'
      else:
        self.pro_method = 'h'
      self.refresh()

    self.register_a_shortcut(
      'space', lambda: self.flip('view_delta'), 'Toggle `view_delta`')
    self.register_a_shortcut(
      'v', modify_view, 'Change the Viewpoints')
    self.register_a_shortcut(
      'm', lambda: self.flip('show_metric'), 'Turn on/off title')
    self.register_a_shortcut(
      'S', lambda: self.flip('show_slice_metric'),
      'Turn on/off `show_slice_metric` option')
    self.register_a_shortcut(
      'w', lambda: self.flip('show_weight_map'), 'Toggle `show_weight_map`')
    self.register_a_shortcut(
      's', lambda: self.flip('show_suv_metric'),
      'Turn on/off `show_suv_metric` option')
    self.register_a_shortcut(
      'A', show_profile,
      'Turn on/off `show_profile` option')
    self.register_a_shortcut(
      'a', switch_pro_method,
      'switch profile\'s method')
    self.register_a_shortcut(
      'z', lambda: pro_move(-1),
      'profile up')
    self.register_a_shortcut(
      'x', lambda: pro_move(1),
      'profile down')

  # endregion: Shortcuts




if __name__ == '__main__':
  pass
