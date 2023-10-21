from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase

import tkinter as tk



class ImageSwitchPanel(FrameBase):
  def __init__(self, master, bg='lightgray', name='frame:image_switch_panel'):
    super().__init__(master=master, bg=bg, name=name)


  def _init_panel(self):
    # (1) Button Function Setting

    # (2) Widgets Setting
    # region: Frame Setting
    frame_patient = tk.Frame(
      self, bg='lightgray', name='frame:patient')
    frame_channel = tk.Frame(
      self, bg='lightgray', name='frame:channel')

    # endregion: Frame Setting

    # region: Button Setting
    button_next_patient = tk.Button(
      frame_patient, text="Next", width=self.width, height=self.height,
      command=lambda: self.main_canvas._set_patient_cursor(1),
      name='button:next patient')

    button_pre_patient = tk.Button(
      frame_patient, text="Pre", width=self.width, height=self.height,
      command=lambda: self.main_canvas._set_patient_cursor(-1),
      name='button:pre patient')

    button_next_channel = tk.Button(
      frame_channel, text="Next", width=self.width, height=self.height,
      command=lambda: self.main_canvas.set_cursor(
        self.main_canvas.Keys.LAYERS, 1, refresh=True),
      name='button:next channel')

    button_pre_channel = tk.Button(
      frame_channel, text="Pre", width=self.width, height=self.height,
      command=lambda: self.main_canvas.set_cursor(
        self.main_canvas.Keys.LAYERS, -1, refresh=True),
      name='button:pre channel')

    # endregion: Button Setting

    # region: Label Setting
    self.text_var['patient'] = tk.StringVar()
    label_patient = tk.Label(
      frame_patient, textvariable=self.text_var['patient'],
      width=self.width * 2, bg='lightgray', name='label:patient')

    self.text_var['channel'] = tk.StringVar()
    label_channel = tk.Label(
      frame_channel, textvariable=self.text_var['channel'],
      width=self.width * 2, bg='lightgray', name='label:channel')
    # endregion: Label Setting

    # (3) Position Setting in Frame
    # region: Position Setting of frame_patient
    self.position_setting(label_patient, [0, 0, 2], self.sticky)
    self.position_setting(button_next_patient, [0, 2, 1], self.sticky)
    self.position_setting(button_pre_patient, [0, 3, 1], self.sticky)

    # endregion: Position Setting of frame_patient

    # region: Position Setting of frame_channel
    self.position_setting(label_channel, [0, 0, 2], self.sticky)
    self.position_setting(button_next_channel, [0, 2, 1], self.sticky)
    self.position_setting(button_pre_channel, [0, 3, 1], self.sticky)

    # endregion: Position Setting of frame_channel



if __name__ == '__main__':
  pass