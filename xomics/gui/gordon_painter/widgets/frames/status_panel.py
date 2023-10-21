from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase

import tkinter as tk



class StatusPanel(FrameBase):
  def __init__(self, master, bg='lightgray', name='frame:status_panel'):
    super().__init__(master=master, bg=bg, name=name)

    self.word_num = 20


  def _init_panel(self):
    # (1) Button Function Setting

    # (2) Widgets Setting
    # region: Frame Setting
    frame_textbox = tk.Frame(
      self, bg=self['background'], name='frame:textbox')

    # endregion: Frame Setting

    # region: TextBox Setting
    text_status = tk.Text(
      frame_textbox, width=self.width * 4, name='text:status', wrap=tk.WORD,
      bg=self['background'])
    text_status.config(state=tk.DISABLED)

    # endregion: TextBox Setting

    # (3) Position Setting in Frame
    # region: Position Setting of frame_textbox
    self.position_setting(text_status, [0, 0, 4], self.sticky)

    # endregion: Position Setting of frame_patient


  def display_in_status_box(self, text):
    word_num = self.word_num

    textbox = self.children['frame:textbox'].children['text:status']
    textbox.config(state=tk.NORMAL)
    textbox.insert(tk.END, '>>' + text + "\n")
    textbox.config(state=tk.DISABLED)

    # Check that the number of lines in the text box is more than 10,
    # and if so, delete the oldest line
    lines = textbox.get(1.0, tk.END).split('\n')
    if len(lines) > word_num:
      textbox.config(state=tk.NORMAL)
      textbox.delete(1.0, 2.0)
      textbox.config(state=tk.DISABLED)



if __name__ == '__main__':
  pass