from collections import OrderedDict

import tkinter as tk



class FrameBase(tk.Frame):
  def __init__(self, master, name, bg='lightgray'):
    super().__init__(master=master, bg=bg, name=name)

    # (1) Parameter Setting

    # (2) Initialization
    self._init_panel()

  # region: Properties
  @property
  def main_canvas(self):
    return self.master.main_canvas

  @property
  def width(self):
    return self.master.width

  @property
  def height(self):
    return self.master.height

  @property
  def sticky(self):
    return self.master.sticky

  @property
  def text_var(self):
    return self.master.text_var

  @property
  def slice_view(self):
    return self.master.slice_view

  @property
  def unit_position(self):
    return self.master.unit_position

  @property
  def bg(self):
    return self['background']

  # endregion: Properties

  def _init_panel(self):
    pass


  def position_setting(self, element, distribution, sticky):
    row, column, columnspan = distribution
    element.grid(
      row=row, column=column, columnspan=columnspan, sticky=sticky)


  def set_position(self):
    for i, f in enumerate(self.children.values()):
      self.unit_position[self.winfo_name()].update({
        f.winfo_name(): [i, 0, 1]
      })


  def refresh(self):
    pass



if __name__ == '__main__':
  pass