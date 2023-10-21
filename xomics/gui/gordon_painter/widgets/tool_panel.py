from collections import OrderedDict
from xomics.gui.gordon_painter.widgets.frames.image_switch_panel import ImageSwitchPanel
from xomics.gui.gordon_painter.widgets.frames.file_process_panel import FileProcessPanel
from xomics.gui.gordon_painter.widgets.frames.annotation_process_panel import AnnotationProcessPanel
from xomics.gui.gordon_painter.widgets.frames.status_panel import StatusPanel

import tkinter as tk



class ToolPanel(tk.Frame):
  def __init__(self, master, bg='lightgray'):
    super().__init__(master=master, bg=bg)

    # Parameter Setting
    self.unit_position = OrderedDict()

    # Create panels
    self.image_switch_panel = ImageSwitchPanel(self, bg='lightgray')
    self.file_process_panel = FileProcessPanel(self, bg='gray')
    self.annotation_process_panel = AnnotationProcessPanel(
      self, bg='lightgray')
    self.status_panel = StatusPanel(self, bg='gray')

    # Initialization
    self._init_tool_panel()


  # region: Properties
  @property
  def main_canvas(self):
    for key in self.master.children.keys():
      if 'gordonpainter' in key:
        return self.master.children[key]

  @property
  def text_var(self):
    return self.main_canvas.context['text_var']

  @property
  def slice_view(self):
    return self.main_canvas.slice_view

  @property
  def width(self):
    return self.main_canvas.style_sheet['unit_width']

  @property
  def height(self):
    return self.main_canvas.style_sheet['unit_height']

  @property
  def sticky(self):
    return self.main_canvas.style_sheet['sticky']

  @property
  def percentile_step(self):
    return self.main_canvas.configs['percentile_step']

  # endregion: Properties

  # region: Init Func
  def _init_tool_panel(self):
    self._init_layout()
    self._init_position_dict()

    self.image_switch_panel.set_position()
    self.file_process_panel.set_position()
    self.annotation_process_panel.set_position()
    self.status_panel.set_position()


  def _init_layout(self):
    for p in self.children.values():
      if 'frame' in p.widgetName:
        p.pack(side='top', fill='both', expand=True)


  def _init_position_dict(self):
    for p in self.children.values():
      self.unit_position.update({
        p.winfo_name(): OrderedDict()
      })


  def refresh(self):
    for f in self.children.values():
      f.refresh()



if __name__ == '__main__':
  pass