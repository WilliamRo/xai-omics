from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os
import re



class DrGordon(Pictor):

  class Keys(Pictor.Keys):
    PATIENTS = 'PaTiEnTs'
    LAYERS = 'LaYeRs'

  def __init__(self, medical_images, title='Dr. Gordon'):
    super(DrGordon, self).__init__(title)

    self.slice_view: SliceView = self.add_plotter(SliceView(self))

    # Create dimensions for different patients
    self.create_dimension(self.Keys.PATIENTS)
    self.create_dimension(self.Keys.LAYERS)

    self.set_data(medical_images)

    self.shortcuts.register_key_event(
      ['J'], lambda: self._set_patient_cursor(1),
      description='Next patient', color='yellow')
    self.shortcuts.register_key_event(
      ['K'], lambda: self._set_patient_cursor(-1),
      description='Previous patient', color='yellow')

    self.shortcuts.register_key_event(
      ['N'], lambda: self.set_cursor(self.Keys.LAYERS, 1, refresh=True),
      description='Next layer', color='yellow')
    self.shortcuts.register_key_event(
      ['P'], lambda: self.set_cursor(self.Keys.LAYERS, -1, refresh=True),
      description='Previous layer', color='yellow')

  # region: Properties

  @property
  def objects(self):
    return self.axes[self.Keys.OBJECTS]

  @objects.setter
  def objects(self, value):
    raise AssertionError('Can not set objects in this way')

  # endregion: Properties

  # region: Private Methods
  def export_patients(self, fps: float = 10, cursor_range=None):
    from tkinter import filedialog
    dir = filedialog.askdirectory()

    # Find cursor range
    if cursor_range is None:
      begin, end = 1, len(self.axes[self.Keys.PATIENTS])
    else:
      if re.match('^\d+:\d+$', cursor_range) is None:
        raise ValueError(
          '!! Illegal cursor range `{}`'.format(cursor_range))
      begin, end = [int(n) for n in cursor_range.split(':')]
    begin, end = max(1, begin), min(len(self.axes[self.Keys.PATIENTS]), end)

    self.cursors[self.Keys.PATIENTS] = begin - 1
    for i in range(begin, end + 1):
      match = re.search(r'PID: (\S+)', self.axes[self.Keys.PATIENTS][i - 1].key)
      filename = f'{i}-' + match.group(1) if match else f'{i}'
      path = os.path.join(dir, filename + '.gif')

      self.animate(fps=fps, path=path)
      self._set_patient_cursor(1)


  def _set_patient_cursor(self, step):
    prev_mi: MedicalImage = self.get_element(self.Keys.PATIENTS)
    self.set_cursor(self.Keys.PATIENTS, step, refresh=False)
    curr_mi: MedicalImage = self.get_element(self.Keys.PATIENTS)

    # Refresh cursors if necessary
    self.refresh_patient(prev_mi.num_slices != curr_mi.num_slices,
                         prev_mi.num_layers != curr_mi.num_layers)
    self.refresh()

  # endregion: Private Methods

  # region: Public Methods

  def set_data(self, medical_images):
    if medical_images is None: return
    self.set_to_axis(self.Keys.PATIENTS, medical_images, overwrite=True)
    self.refresh_patient()

  def refresh_patient(self, refresh_slice=True, refresh_layer=True):
    mi: MedicalImage = self.get_element(self.Keys.PATIENTS)
    if refresh_slice: self.set_to_axis(
      self.Keys.OBJECTS, list(range(mi.num_slices)), overwrite=True)
    if refresh_layer: self.set_to_axis(
      self.Keys.LAYERS, list(range(mi.num_layers)), overwrite=True)

  # endregion: Public Methods

  # region: Overwritting

  def refresh(self, wait_for_idle=False):
    self.title_suffix = f' - {self.slice_view.displayed_layer_key}'
    super().refresh(wait_for_idle)

  ep = export_patients
  # endregion: Overwritting



class SliceView(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(SliceView, self).__init__(self.view_slice, pictor)

    self.annotations_to_show = []

    # Settable attributes
    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('ctv_key', None, str, 'Key of CTV to show')
    self.new_settable_attr('cmap', 'gray', str, 'Color map')
    self.new_settable_attr('vmin', 0., float, 'Min value')
    self.new_settable_attr('vmax', 1., float, 'Max value')
    self.new_settable_attr('show_ground_truth', False, bool,
                           'whether to show the ground truth')
    self.new_settable_attr('show_prediction', False, bool,
                           'whether to show the prediction')

  # region: Properties

  @property
  def selected_medical_image(self) -> MedicalImage:
    return self.pictor.get_element(self.pictor.Keys.PATIENTS)

  @property
  def displayed_layer_key(self):
    i_layer = self.pictor.get_element(self.pictor.Keys.LAYERS)
    return list(self.selected_medical_image.images.keys())[i_layer]

  # endregion: Properties

  # region: Plot

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

  # endregion: Plot

  # region: Overwritting

  def register_shortcuts(self):
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')

    def flip_anno():
      key_list = list(self.selected_medical_image.labels.keys())
      if len(key_list) == 0: return
      key = key_list[0]
      if key in self.annotations_to_show: self.annotations_to_show.remove(key)
      else: self.annotations_to_show.append(key)
      self.refresh()

    self.register_a_shortcut('A', flip_anno, 'Toggle first annotation')

  def register_to_master(self, pictor):
    super().register_to_master(pictor)

    def get_anno_hints():
      hints = ['Annotations', '-' * 11]
      hints += [f'[{i + 1}] {k}' for i, k in enumerate(
        self.selected_medical_image.labels.keys())]
      return '\n'.join(hints)

    self.command_hints['ta'] = get_anno_hints

  # endregion: Overwritting

  # region: Commands

  def toggle_annotation(self, indices: str = None, auto_refresh=True):
    if indices is None:
      self.annotations_to_show = []
    else:
      indices = [int(str_i) - 1 for str_i in indices.split(',')]
      anno_keys = list(self.selected_medical_image.labels.keys())
      self.annotations_to_show = [anno_keys[i] for i in indices]

    if auto_refresh: self.refresh()

  ta = toggle_annotation

  # endregion: Commands



def load_npy_file(file_path):
  return np.load(file_path, allow_pickle=True).tolist()



if __name__ == '__main__':
  input_dir = r'../../data/00-CT-demo/'
  ct_file = 'demo1_ct.npy'
  label_file = 'demo1_label.npy'

  # Loading data
  # ct and label have the same shape
  # [patient, slice, H, W]
  ct = load_npy_file(os.path.join(input_dir, ct_file))
  label = load_npy_file(os.path.join(input_dir, label_file))

  # Normalization
  ct = np.squeeze(ct) / np.max(ct)
  label = np.squeeze(label) / np.max(label)

  mi_1 = MedicalImage('Patient-1', {'CT': ct}, {'Label-1': label})
  mi_2 = MedicalImage('Patient-2', {'CT': ct[3:]}, {'Label-1': label[3:]})

  # Visualization
  dg = DrGordon([mi_1, mi_2])
  dg.show()
