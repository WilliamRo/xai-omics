from xomics.gui.dr_gordon import DrGordon, SliceView
from xomics import MedicalImage
from tkinter import Menu

import matplotlib.pyplot as plt
import numpy as np
import os


class GordonPainter(DrGordon):

  def __init__(
      self, medical_images, title='Gordon Painter', figure_size=(10, 10)):
    super().__init__(medical_images, title=title, figure_size=figure_size)
    self.slice_view = GordonView(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.slice_view], overwrite=True)
    self.create_menu_bar()


  def refresh(self, wait_for_idle=False):
    self.refresh_annotation()
    super().refresh()


  def refresh_annotation(self):
    selected_mi = self.get_element(self.Keys.PATIENTS)
    label_set = set(selected_mi.labels.keys())
    annotation_set = set(self.slice_view.annotations_to_show)

    if not annotation_set.issubset(label_set):
      element_not_in_label = annotation_set - label_set
      new_annotation = [a for a in self.slice_view.annotations_to_show
                        if a not in element_not_in_label]
      self.slice_view.annotations_to_show = new_annotation


  def create_menu_bar(self):
    menu_bar = Menu(self.master)
    self.master.configure(menu=menu_bar)

    # create File menu
    file_menu = Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # add the menu items to the file menu
    file_menu.add_command(label="Open file")
    file_menu.add_command(label="Save segmentation")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=self.master.quit)

    # create Edit menu
    # edit_menu = Menu(menu_bar, tearoff=0)
    # menu_bar.add_cascade(label="Edit", menu=edit_menu)
    #
    # # add menu items to the edit menu
    # edit_menu.add_command(label="Cut")
    # edit_menu.add_command(label="Copy")
    # edit_menu.add_command(label="Paste")



class GordonView(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)

    # Settable attributes
    self.new_settable_attr('painter', False, bool, 'Painter')
    self.pictor.canvas._canvas.mpl_connect(
      'button_press_event', self.on_mouse_click)

  def register_shortcuts(self):
    super().register_shortcuts()

    self.register_a_shortcut(
      'Q', lambda: self.flip('painter'), 'Turn on/off painter')


  def on_mouse_click(self, event):
    if event.xdata is not None and event.ydata is not None and self.get('painter'):
      # Coordinates of the mouse click
      y, x = int(event.xdata), int(event.ydata)
      num_slice = self.pictor.cursors[self.pictor.Keys.OBJECTS]
      image = self.selected_medical_image.images[self.displayed_layer_key]
      data = image[num_slice, x, y]

      # get the percentile
      flatten_image = np.sort(image.flatten())
      percentile = round(
        ((np.searchsorted(flatten_image, data) + 1) / flatten_image.size * 100), 2)

      # calculate the percentile and the threshold
      percentile = max(percentile - 0.5, 0)
      threshold = np.percentile(image, q=percentile)

      # get the mask and add to mi
      mask = (image >= threshold).astype(np.uint8)
      mask = self.amend_mask(mask, tuple([num_slice, x, y]))
      self.selected_medical_image.labels['mask'] = mask

      # visualiza the new label
      self.annotations_to_show.append('mask')
      self.refresh()

      print(f"Mouse clicked at ({x:.2f}, {y:.2f})")
      print(f"percentile: {percentile}")
      print(f"threshold: {threshold}")


  def amend_mask(self, mask: np.ndarray, chosen_data_address: tuple):
    from scipy.ndimage import label, generate_binary_structure

    structure = generate_binary_structure(3, 1)
    # Get connected region for denoise
    labeled_image, num_features = label(mask, structure)

    # Gets the label number of the connected area where the mouse clicks
    num_label = labeled_image[chosen_data_address]
    labeled_image[labeled_image != num_label] = 0
    labeled_image[labeled_image == num_label] = 1

    return labeled_image



if __name__ == '__main__':
  data_dir = r'../../data/02-PET-CT-Y1/results/mi/'

  mi_file = os.listdir(data_dir)

  mi_list = []
  for file in mi_file:
    mi_list.append(MedicalImage.load(os.path.join(data_dir, file)))

  # Visualization
  gp = GordonPainter(mi_list)
  gp.slice_view.set('vmin', auto_refresh=False)
  gp.slice_view.set('vmax', auto_refresh=False)
  gp.show()

