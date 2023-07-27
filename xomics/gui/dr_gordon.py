from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np
import os



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

  # endregion: Overwritting



class SliceView(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(SliceView, self).__init__(self.view_slice, pictor)

    # Settable attributes
    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('ctv_key', None, str, 'Key of CTV to show')
    self.new_settable_attr('cmap', 'gray', float, 'Color map')
    self.new_settable_attr('vmin', 0., float, 'Min value')
    self.new_settable_attr('vmax', 1., float, 'Max value')
    self.new_settable_attr('show_ground_truth', False, bool,
                           'whether to show the ground truth')
    self.new_settable_attr('show_prediction', False, bool,
                           'whether to show the prediction')

  @property
  def selected_medical_image(self) -> MedicalImage:
    return self.pictor.get_element(self.pictor.Keys.PATIENTS)

  @property
  def displayed_layer_key(self):
    i_layer = self.pictor.get_element(self.pictor.Keys.LAYERS)
    return list(self.selected_medical_image.images.keys())[i_layer]

  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    label_key = 'Label-1'
    mi: MedicalImage = self.selected_medical_image

    # Show slice
    im = ax.imshow(mi.images[self.displayed_layer_key][x],
                   cmap=self.get('cmap'), vmin=self.get('vmin'),
                   vmax=self.get('vmax'))

    # TODO: to be refactored
    if self.get('show_ground_truth') and label_key in mi.labels:
      mask = mi.labels[label_key][x]
      img = np.zeros((512, 512, 4), dtype=np.float32)
      img[..., 0] = 1.0
      img[..., 3] = mask
      ax.imshow(img, alpha=0.2)

    if self.get('show_prediction') and 'Prediction' in mi.labels:
      mask = mi.labels['Prediction'][x]
      img = np.zeros((512, 512, 4), dtype=np.float32)
      img[..., 2] = 1.0
      img[..., 3] = mask
      ax.imshow(img, alpha=0.2)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def register_shortcuts(self):
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')

    self.register_a_shortcut(
      'Q', lambda: self.flip('show_ground_truth'),
      'whether to show the ground truth')

    self.register_a_shortcut(
      'W', lambda: self.flip('show_prediction'),
      'whether to show the prediction')



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
