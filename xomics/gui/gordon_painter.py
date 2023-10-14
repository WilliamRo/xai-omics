from xomics.gui.dr_gordon import DrGordon, SliceView
from xomics import MedicalImage
from tqdm import tqdm

import tkinter as tk
import numpy as np
import os


class GordonPainter(DrGordon):

  def __init__(
      self, medical_images, title='Gordon Painter', figure_size=(10, 10)):
    super().__init__(medical_images, title=title, figure_size=figure_size)

    self.text_var_percentile = None
    self.button_width = 16
    self.button_height = 2
    self.percentile_step = 0.05

    self.pack(side='left', fill='both', expand=True)
    self.slice_view = GordonView(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.slice_view], overwrite=True)

    self.create_menu_bar()
    self.create_tool_bar()


  def refresh(self, wait_for_idle=False):
    self.refresh_annotation()
    self.refresh_text()
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


  def refresh_text(self):
    if self.slice_view.percentile is not None:
      self.text_var_percentile.set(
        f'Percentile: {str(round(self.slice_view.percentile, 2))}')


  def create_menu_bar(self):
    menu_bar = tk.Menu(self.master)
    self.master.configure(menu=menu_bar)

    # create File menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # add the menu items to the file menu
    file_menu.add_command(label="Open DICOM file",
                          command=self.open_dicom_file)
    file_menu.add_command(label="Open NIfTI file",
                          command=self.open_nifti_file)
    file_menu.add_command(label="Open MI file",
                          command=self.open_mi_file)
    file_menu.add_separator()

    file_menu.add_command(label="Save segmentation as NIfTI",
                          command=self.save_segmentation_as_nifti)
    file_menu.add_command(label="Save segmentation as MI",
                          command=self.save_segmentation_as_mi)
    file_menu.add_separator()

    file_menu.add_command(label="Exit", command=self.master.quit)


  def create_tool_bar(self):
    tool_bar = tk.Frame(self.master, bg='lightgray')
    tool_bar.pack(side='right', fill='both', expand=True)

    # region: Button Setting
    width, height = self.button_width, self.button_height
    percentile_step = self.percentile_step

    button_painter = tk.Button(
      tool_bar, text="Painter", width=2 * width, height=height,
      command=lambda: self.slice_view.flip('painter'))

    button_next_patient = tk.Button(
      tool_bar, text="Next Patient", width=width, height=height,
      command=lambda: self._set_patient_cursor(1))

    button_pre_patient = tk.Button(
      tool_bar, text="Pre Patient", width=width, height=height,
      command=lambda: self._set_patient_cursor(-1))

    button_next_channel = tk.Button(
      tool_bar, text="Next Channel", width=width, height=height,
      command=lambda: self.set_cursor(self.Keys.LAYERS, 1, refresh=True))

    button_pre_channel = tk.Button(
      tool_bar, text="Pre Channel", width=width, height=height,
      command=lambda: self.set_cursor(self.Keys.LAYERS, -1, refresh=True))

    button_add_percentile = tk.Button(
      tool_bar, text="Add Percentile", width=width, height=height,
      command=lambda: self.button_set_percentile(percentile_step))

    button_sub_percentile = tk.Button(
      tool_bar, text="Sub Percentile", width=width, height=height,
      command=lambda: self.button_set_percentile(-percentile_step))

    # endregion: Button Setting

    # region: Text Setting
    self.text_var_percentile = tk.StringVar()
    self.text_var_percentile.set('')
    text_percentile = tk.Label(
      tool_bar, textvariable=self.text_var_percentile, width=2 * width)

    # endregion: Text Setting

    # region: Layout Setting
    button_painter.grid(row=0, column=0, columnspan=2)
    button_next_patient.grid(row=1, column=0)
    button_pre_patient.grid(row=1, column=1)
    button_next_channel.grid(row=2, column=0)
    button_pre_channel.grid(row=2, column=1)
    button_add_percentile.grid(row=3, column=0)
    button_sub_percentile.grid(row=3, column=1)
    text_percentile.grid(row=4, column=0, columnspan=2)

    # endregion: Layout Setting


  # region: Open func and Save func
  def open_dicom_file(self):
    dir_path = tk.filedialog.askdirectory(title="选择DICOM文件夹")
    if not dir_path: return


  def open_nifti_file(self):
    file_paths = tk.filedialog.askopenfilenames(title="选择NIfTI文件")
    if not file_paths: return


  def open_mi_file(self):
    file_paths = tk.filedialog.askopenfilenames(
      title="选择MI文件", filetypes=[('MI文件', '*.mi')])
    if not file_paths: return

    mi_list = [MedicalImage.load(f) for f in file_paths]
    self.set_data(mi_list)
    self.refresh()


  def save_segmentation_as_nifti(self):
    print("save segmentation")


  def save_segmentation_as_mi(self):
    dir_path = tk.filedialog.askdirectory()
    if not dir_path: return

    for mi in tqdm(self.axes[self.Keys.PATIENTS], desc='Saving mi file'):
      file_name = mi.key.split(' ')[0] + '.mi'
      mi.save(os.path.join(dir_path, file_name))

    print(f'Successfully saved {len(self.axes[self.Keys.PATIENTS])} samples')
    print(f"Data saved to {dir_path}")

  # endregion: open and save

  # region: Button func
  def button_set_percentile(self, step):
    if self.slice_view.percentile and self.slice_view.click_address:
      self.slice_view.percentile = self.slice_view.percentile + step

      # restricted range
      self.slice_view.percentile = min(100.00, self.slice_view.percentile)
      self.slice_view.percentile = max(0.00, self.slice_view.percentile)

      self.slice_view.adjust_mask(self.slice_view.percentile)
      self.refresh()

  # endregion: Button func


class GordonView(SliceView):

  def __init__(self, gordon_painter=None):
    # Call parent's constructor
    super().__init__(pictor=gordon_painter)

    self.percentile = None
    self.click_address = None

    # Settable attributes
    self.new_settable_attr('painter', False, bool, 'Painter')
    self.pictor.canvas._canvas.mpl_connect(
      'button_press_event', self.on_mouse_click)
    self.pictor.canvas._canvas.mpl_connect(
      'scroll_event', self.on_mouse_scroll)


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
      self.percentile = percentile
      self.click_address = tuple([num_slice, x, y])
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


  def on_mouse_scroll(self, event):
    if event.button == 'up':
      self.pictor.set_cursor(self.pictor.Keys.OBJECTS, 1, refresh=True)
    elif event.button == 'down':
      self.pictor.set_cursor(self.pictor.Keys.OBJECTS, -1, refresh=True)


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


  def adjust_mask(self, percentile):
    image = self.selected_medical_image.images[self.displayed_layer_key]
    threshold = np.percentile(image, q=percentile)

    new_mask = (image >= threshold).astype(np.uint8)
    new_mask = self.amend_mask(new_mask, self.click_address)
    self.selected_medical_image.labels['mask'] = new_mask


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

