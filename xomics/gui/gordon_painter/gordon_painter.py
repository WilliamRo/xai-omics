from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage
from tqdm import tqdm
from collections import OrderedDict
from widgets.menu_bar import MenuBar
from widgets.tool_panel import ToolPanel
from widgets.gordon_view import GordonView
from tkinter import messagebox

import tkinter as tk
import windnd
import numpy as np
import os



class GordonPainter(DrGordon):

  def __init__(
      self, medical_images, title='Gordon Painter', figure_size=(10, 10)):
    # Call parent's constructor, add (1) canvas
    super().__init__(medical_images, title=title, figure_size=figure_size)

    windnd.hook_dropfiles(self.master, func=self.dragged_files)
    self.slice_view = GordonView(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.slice_view], overwrite=True)

    # (2) Menu bar
    self.menu_bar = MenuBar(self.master)
    self.master.configure(menu=self.menu_bar)

    # (3) Tool panel
    self.tool_panel = ToolPanel(self.master, bg='lightgray')

    # (4) Initialize layout
    self._init_layout()


  # region: Properties

  @DrGordon.property()
  def context(self): return {
    'text_var': OrderedDict(),
    'unit_position': OrderedDict()
  }

  @DrGordon.property()
  def style_sheet(self): return {
    'unit_width': 10,
    'unit_height': 2,
    'sticky': 'nsew'
  }

  @DrGordon.property()
  def configs(self): return {
    'percentile_step': 0.05
  }

  @property
  def patients(self):
    return self.axes[self.Keys.PATIENTS]

  @property
  def channels(self):
    return self.axes[self.Keys.LAYERS]

  @property
  def text_var(self):
    return self.context['text_var']

  @property
  def unit_position(self):
    return self.context['unit_position']

  # endregion: Properties

  # region: Style

  def _init_layout(self):
    self.pack(side='left', fill='both', expand=True)
    self.tool_panel.pack(side='right', fill='both', expand=True)

  # endregion: Style

  def refresh(self, wait_for_idle=False):
    if self.patients == []: return

    self._refresh_text()
    self._refresh_button()
    self._refresh_position()
    self._refresh_annotation()
    super().refresh()


  def _refresh_annotation(self):
    selected_mi = self.get_element(self.Keys.PATIENTS)
    label_set = set(selected_mi.labels.keys())
    annotation_set = set(self.slice_view.annotations_to_show)

    if not annotation_set.issubset(label_set):
      element_not_in_label = annotation_set - label_set
      new_annotation = [a for a in self.slice_view.annotations_to_show
                        if a not in element_not_in_label]
      self.slice_view.annotations_to_show = new_annotation


  def _refresh_text(self):
    # (1) image switch panel
    # region: Text Variable
    self.text_var['patient'].set(
      f'Patient: {self.slice_view.selected_medical_image.key.split("_")[-1]}')
    self.text_var['channel'].set(
      f'Channel: {self.slice_view.displayed_layer_key.upper()}')
    # endregion: Text Variable

    # (2) annotation process panel
    # region: label:percentile
    if self.slice_view.percentile and self.slice_view.get('painter'):
      self.text_var['percentile'].set(
        f'Percentile: {str(round(self.slice_view.percentile, 2))}')
    else:
      self.text_var['percentile'].set('Percentile: None')
    # endregion: label:percentile

    # region: Dynamic Label Frame

    # endregion: Dynamic Label Frame


  def _refresh_button(self):
    # (1) annotation process panel
    # region: Button show and Button delete in dynamic button
    label_list = list(self.slice_view.selected_medical_image.labels.keys())
    units = list(self.tool_panel.annotation_process_panel.children.values())
    showed_label_list = [u.winfo_name().split(':label-')[-1]
                         for u in units if 'frame:label' in u.winfo_name()]
    label_set, showed_label_set = set(label_list), set(showed_label_list)
    if label_set != showed_label_set:
      for l in showed_label_set:
        self.tool_panel.annotation_process_panel.children[f'frame:label-{l}'].destroy()
        del self.tool_panel.unit_position[
          self.tool_panel.annotation_process_panel.winfo_name()][f'frame:label-{l}']
      for i, l in enumerate(label_set):
        self.tool_panel.create_dynamic_label_frame(l, i + 1)
    # endregion: Button show and Button delete

    # TODO ：
    #  The boundaries of the buttons are not clear
    element_button = self.find_widgets_in_frame(
      self.tool_panel.annotation_process_panel, tk.Button)
    element_button = [
      b for b in element_button if 'painter' not in b.winfo_name()]
    state_1 = 'active' if self.slice_view.get('painter') else 'disable'
    state_2 = 'disable' if self.slice_view.get('painter') else 'active'

    for b in element_button:
      if 'percentile' in b.winfo_name():
        b.configure(state=state_1)
      else:
        b.configure(state=state_2)


  def _refresh_position(self):
    bars = list(self.tool_panel.children.values())
    position_dict = self.tool_panel.unit_position

    for b in bars:
      for name in position_dict[b.winfo_name()].keys():
        unit = b.children[name]
        row, column, columnspan = position_dict[b.winfo_name()][name]
        unit.grid(
          row=row, column=column, columnspan=columnspan, sticky='nsew')


  def dragged_files(self, files):
    files = [f.decode('gbk', 'ignore') for f in files]

    # (1) Determine the file format
    file_format = files[0].split('.')[-1]
    if file_format not in ['dcm', 'DCM', 'nii', 'gz', 'mi']:
      messagebox.showerror('Error', 'Please enter the correct file format!')
      return

    # (2) file format == 'mi'
    if all(file.endswith(".mi") for file in files):
      self.menu_bar.open_mi_file(files)

    # (3) file format == 'nii' or 'nii.gz'
    if all(file.endswith(".nii") or file.endswith(".nii.gz") for file in files):
      self.menu_bar.open_nifti_file(files[0])

    # (4) file format == 'dcm'
    if all(file.endswith(".dcm") or file.endswith(".DCM") for file in files):
      self.menu_bar.open_dicom_file(files[0])


  def find_widgets_in_frame(self, frame, type):
    button_instances = []
    for widget in frame.winfo_children():
      if isinstance(widget, type):
        button_instances.append(widget)
      elif isinstance(widget, tk.Frame):
        button_instances.extend(self.find_widgets_in_frame(widget, type))

    return button_instances



if __name__ == '__main__':

  gp = GordonPainter(None)
  gp.slice_view.set('vmin', auto_refresh=False)
  gp.slice_view.set('vmax', auto_refresh=False)
  gp.show()

