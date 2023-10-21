from xomics.gui.dr_gordon import DrGordon
from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase
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
    self.for_entry()


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

    self.tool_panel.refresh()
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


  def for_entry(self):
    def on_window_click(event):
      if on_window_click.click_count == 0:
        on_window_click.click_count = 1
        on_window_click.first_click_time = event.time
        on_window_click.single_click_timer = self.master.after(
          threshold_time, process_single_click, event)
      else:
        self.master.after_cancel(on_window_click.single_click_timer)
        on_window_click.click_count = 0
        time_diff = event.time - on_window_click.first_click_time
        if time_diff < threshold_time: return
        else: process_single_click(event)

    def process_single_click(event):
      on_window_click.click_count = 0
      frame = self.tool_panel
      entry_list = FrameBase.find_widgets_in_frame(frame, tk.Entry)
      if entry_list == []: return
      for e in entry_list:
        if e.grid_info() == {}: continue
        elif event.widget != e:
          e.focus()
          e.event_generate("<Return>")


    self.master.bind("<Button-1>", on_window_click)
    threshold_time = 200
    on_window_click.click_count = 0
    on_window_click.single_click_timer = None
    on_window_click.first_click_time = 0



if __name__ == '__main__':

  gp = GordonPainter(None)
  gp.slice_view.set('vmin', auto_refresh=False)
  gp.slice_view.set('vmax', auto_refresh=False)
  gp.show()

