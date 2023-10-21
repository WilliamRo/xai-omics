from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase
from functools import partial
from tkinter import messagebox

import tkinter as tk



class ImageSwitchPanel(FrameBase):
  def __init__(self, master, bg='lightgray', name='frame:image_switch_panel'):
    super().__init__(master=master, bg=bg, name=name)


  def _init_panel(self):
    # (1) Button Function Setting
    # region: Button Func
    def on_double_click(event, label, entry):
      if self.slice_view.get('painter'): return
      label.grid_forget()
      self.position_setting(entry, [0, 0, 2], self.sticky)
      entry.delete(0, tk.END)
      entry.insert(0, label.cget('text'))
      entry.focus()


    def on_enter(event, label, entry, p_or_c):
      new_name = entry.get()
      self.position_setting(label, [0, 0, 2], self.sticky)
      entry.grid_forget()

      if self.text_var[p_or_c].get() == new_name: return
      elif p_or_c == 'patient':
        self.slice_view.selected_medical_image.key = new_name
      elif new_name in self.slice_view.selected_medical_image.images.keys():
        messagebox.showerror(
          'Error', f'{new_name} already exists, please change the another one.')
        return
      else:
        old_name = self.text_var[p_or_c].get()
        data = self.slice_view.selected_medical_image.images[old_name]
        self.slice_view.selected_medical_image.images[new_name] = data
        del self.slice_view.selected_medical_image.images[old_name]

      self.main_canvas.refresh()


    # endregion: Button Func

    # (2) Widgets Setting
    # region: Frame Setting
    frame_patient = tk.Frame(self, bg=self.bg, name='frame:patient')
    frame_channel = tk.Frame(self, bg=self.bg, name='frame:channel')

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
      width=self.width * 2, bg=self.bg, name='label:patient')

    self.text_var['channel'] = tk.StringVar()
    label_channel = tk.Label(
      frame_channel, textvariable=self.text_var['channel'],
      width=self.width * 2, bg=self.bg, name='label:channel')
    # endregion: Label Setting

    # region: Entry Setting
    entry_patient = tk.Entry(frame_patient, name='entry:patient')
    entry_channel = tk.Entry(frame_channel, name='entry:channel')

    label_patient.bind(
      "<Double-Button-1>", partial(on_double_click, label=label_patient,
                                   entry=entry_patient))
    entry_patient.bind(
      "<Return>", partial(on_enter, label=label_patient,
                          entry=entry_patient, p_or_c='patient'))

    label_channel.bind(
      "<Double-Button-1>", partial(on_double_click, label=label_channel,
                                   entry=entry_channel))
    entry_channel.bind(
      "<Return>", partial(on_enter, label=label_channel,
                          entry=entry_channel, p_or_c='channel'))

    # self.master.bind("<Button-1>", on_window_click)
    # endregion: Entry Setting

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


  def refresh(self):
    # region: (1) Label
    if self.slice_view.selected_medical_image is None: return
    self.text_var['patient'].set(
      self.slice_view.selected_medical_image.key)
    self.text_var['channel'].set(
      self.slice_view.displayed_layer_key)

    # endregion: (1) Label

    super().refresh()



if __name__ == '__main__':
  pass