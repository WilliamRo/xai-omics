from tqdm import tqdm
from collections import OrderedDict
from tkinter import ttk
from functools import partial

import tkinter as tk
import os



class ToolPanel(tk.Frame):
  def __init__(self, master, bg='lightgray'):
    super().__init__(master=master, bg=bg)

    # Parameter Setting
    self.unit_position = OrderedDict()

    # Create panels
    self.image_switch_panel = tk.Frame(
      self, bg='lightgray', name='frame:image switch panel')
    self.file_process_panel = tk.Frame(
      self, bg='lightgray', name='frame:file process panel')
    self.annotation_process_panel = tk.Frame(
      self, bg='gray', name='frame:annotation process panel')
    self.status_panel = tk.Frame(
      self, bg='gray', name='frame:status panel')

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

    self._create_image_switch_panel()
    self._create_file_process_panel()
    self._create_annotation_process_panel()
    self._create_status_panel()


  def _init_layout(self):
    for p in self.children.values():
      if 'frame' in p.widgetName:
        p.pack(side='top', fill='both', expand=True)


  def _init_position_dict(self):
    for p in self.children.values():
      self.unit_position.update({
        p.winfo_name(): OrderedDict()
      })


  def _create_image_switch_panel(self):

    # Frame Setting
    frame_patient = tk.Frame(
      self.image_switch_panel, bg='lightgray', name='frame:patient')
    frame_channel = tk.Frame(
      self.image_switch_panel, bg='lightgray', name='frame:channel')

    # Button Setting
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
        self.main_canvas.Keys.LAYERS, -1,refresh=True),
      name='button:pre channel')

    # Label Setting
    self.text_var['patient'] = tk.StringVar()
    label_patient = tk.Label(
      frame_patient, textvariable=self.text_var['patient'],
      width=self.width * 2, bg='lightgray', name='label:patient')

    self.text_var['channel'] = tk.StringVar()
    label_channel = tk.Label(
      frame_channel, textvariable=self.text_var['channel'],
      width=self.width * 2, bg='lightgray', name='label:channel')

    # Position Setting in Frame
    self.position_setting(label_patient, [0, 0, 2], self.sticky)
    self.position_setting(button_next_patient, [0, 2, 1], self.sticky)
    self.position_setting(button_pre_patient, [0, 3, 1], self.sticky)

    self.position_setting(label_channel, [0, 0, 2], self.sticky)
    self.position_setting(button_next_channel, [0, 2, 1], self.sticky)
    self.position_setting(button_pre_channel, [0, 3, 1], self.sticky)

    # Position Setting out of Frame
    self.unit_position[self.image_switch_panel.winfo_name()].update({
      frame_patient.winfo_name(): [0, 0, 1],
      frame_channel.winfo_name(): [1, 0, 1]
    })


  def _create_file_process_panel(self):
    # Button Func
    def mi_save():
      file_type = [('MI文件', '*.mi')]
      mi = self.slice_view.selected_medical_image
      initial_file = mi.key + '.mi'

      file_path = tk.filedialog.asksaveasfilename(
        filetypes=file_type, initialfile=initial_file)
      if not file_path: return
      mi.save(file_path)
      self.display_in_status_box(f'{file_path} saved successfully.')


    def nii_save():
      dir_path = tk.filedialog.askdirectory()
      if not dir_path: return
      mi = self.slice_view.selected_medical_image
      mi.save_as_nii(dir_path)

      self.display_in_status_box('nii saved successfully.')


    # Frame Setting
    frame_mi_save = tk.Frame(
      self.file_process_panel, bg='lightgray', name='frame:mi save')
    frame_nii_save = tk.Frame(
      self.file_process_panel, bg='lightgray', name='frame:nii save')

    # Button Setting
    button_mi_save = tk.Button(
      frame_mi_save, text="Save as mi", width=self.width * 2,
      height=self.height, command=lambda: mi_save(), name='button:mi save')

    button_nii_save = tk.Button(
      frame_nii_save, text="Save as nii", width=self.width * 2,
      height=self.height, command=lambda: nii_save(), name='button:nii save')

    # Label Setting
    label_mi_save = tk.Label(
      frame_mi_save, text='MI: ', width=self.width * 2,
      bg='lightgray', name='label:mi save')
    label_nii_save = tk.Label(
      frame_nii_save, text='NIfTI: ', width=self.width * 2,
      bg='lightgray', name='label:nii save')

    # Position Setting in Frame
    self.position_setting(label_mi_save, [0, 0, 2], self.sticky)
    self.position_setting(button_mi_save, [0, 2, 2], self.sticky)

    self.position_setting(label_nii_save, [0, 0, 2], self.sticky)
    self.position_setting(button_nii_save, [0, 2, 2], self.sticky)

    # Position Setting out of Frame
    self.unit_position[self.file_process_panel.winfo_name()].update({
      frame_mi_save.winfo_name(): [0, 0, 1],
      frame_nii_save.winfo_name(): [1, 0, 1]
    })


  def _create_annotation_process_panel(self):
    # (1) Button Function Setting
    # region: Button Func
    def button_set_percentile(step):
      if self.slice_view.percentile and self.slice_view.click_address:
        self.slice_view.percentile = self.slice_view.percentile + step

        # restricted range
        self.slice_view.percentile = min(
          100.00, self.slice_view.percentile)
        self.slice_view.percentile = max(0.00, self.slice_view.percentile)

        self.slice_view.adjust_mask(self.slice_view.percentile)
        self.main_canvas.refresh()

    def button_set_painter():
      self.slice_view.flip('painter')
      self.display_in_status_box(
        f'Button Painter set to {self.slice_view.get("painter")}')

    # endregion: Button Func

    # (2) Widgets Setting
    # region: Parameter Setting
    percentile_step = self.percentile_step
    # endregion: Parameter Setting

    # region: Frame Setting
    frame_painter = tk.Frame(
      self.annotation_process_panel, bg='lightgray', name='frame:painter')

    # endregion: Frame Setting

    # region: Button Setting
    button_painter = tk.Button(
      frame_painter, text='Painter', height=self.height,
      command=lambda: button_set_painter(),
      name='button:painter')

    button_add_percentile = tk.Button(
      frame_painter, text="+", width=self.width, height=self.height,
      command=lambda s=percentile_step: button_set_percentile(s),
      name='button:add percentile')

    button_sub_percentile = tk.Button(
      frame_painter, text="-", width=self.width, height=self.height,
      command=lambda s=percentile_step: button_set_percentile(-s),
      name='button:sub percentile')

    # endregion: Button Setting

    # region: Label Setting
    self.text_var['percentile'] = tk.StringVar()
    label_percentile = tk.Label(
      frame_painter, textvariable=self.text_var['percentile'],
      width=self.width * 2, bg='gray', name='label:percentile')

    label_mask_prompt = tk.Label(
      frame_painter, text='All labels are below: ', width=self.width * 4,
      bg='gray', name='label:mask prompt')

    # endregion: Label Setting

    # (3) Position Setting
    # region: Position Setting in Frame
    self.position_setting(button_sub_percentile, [0, 0, 1], self.sticky)
    self.position_setting(label_percentile, [0, 1, 2], self.sticky)
    self.position_setting(button_add_percentile, [0, 3, 1], self.sticky)
    self.position_setting(button_painter, [1, 0, 4], self.sticky)
    self.position_setting(label_mask_prompt, [2, 0, 4], self.sticky)

    # endregion: Position Setting in Frame

    # region: Position Setting out of Frame
    self.unit_position[self.annotation_process_panel.winfo_name()].update({
      frame_painter.winfo_name(): [0, 0, 1]
    })

    # endregion: Position Setting out of Frame

    # (4) Dynamic widgets Setting
    # region: Dynamic Button Setting
    for i, layer in enumerate(self.main_canvas.channels):
      channel_name = list(
        self.slice_view.selected_medical_image.labels.keys())[layer]

      self.create_dynamic_label_frame(channel_name, i + 1)

    # endregion: Dynamic Button Setting


  def _create_status_panel(self):

    # Frame Setting
    frame_textbox = tk.Frame(
      self.status_panel, bg=self.status_panel['background'],
      name='frame:textbox')

    # TextBox Setting
    text_status = tk.Text(
      frame_textbox, width=self.width * 4, name='text:status', wrap=tk.WORD,
      bg=self.status_panel['background'])
    text_status.config(state=tk.DISABLED)

    # Position Setting in Frame
    self.position_setting(text_status, [0, 0, 4], self.sticky)

    # Position Setting out of Frame
    self.unit_position[self.status_panel.winfo_name()].update({
      frame_textbox.winfo_name(): [0, 0, 1]
    })


  # endregion: Init Func

  def position_setting(self, element, distribution, sticky):
    row, column, columnspan = distribution
    element.grid(
      row=row, column=column, columnspan=columnspan, sticky=sticky)


  def create_dynamic_label_frame(self, channel_name, row):
    # (1) Button Function Setting
    # region: Button Func
    def button_show_label(label_name):
      if label_name in self.slice_view.annotations_to_show:
        self.slice_view.annotations_to_show.remove(label_name)
        self.display_in_status_box(f'{label_name} hided.')
      else:
        self.slice_view.annotations_to_show.append(label_name)
        self.display_in_status_box(f'{label_name} showed.')

      self.main_canvas.refresh()


    def button_delete_label(label_name):
      result = tk.messagebox.askquestion(
        'Confirm', f'Confirm to delete label "{label_name}"')

      if result == 'yes':
        del self.slice_view.selected_medical_image.labels[label_name]
        self.main_canvas.refresh()


    def on_double_click(event, label, entry):
      if self.slice_view.get('painter'): return
      label.grid_forget()
      self.position_setting(entry, [0, 0, 2], self.sticky)
      entry.delete(0, tk.END)
      entry.insert(0, label.cget('text'))
      entry.focus()


    def on_enter(event, label, entry, channel_name):
      new_channel_name = entry.get()

      # update selected_medical_image.labels
      if channel_name != new_channel_name:
        label_data = self.slice_view.selected_medical_image.labels[channel_name]
        self.slice_view.selected_medical_image.labels[new_channel_name] = label_data
        del self.slice_view.selected_medical_image.labels[channel_name]
        self.display_in_status_box(
          f'rename label-{channel_name} to label-{new_channel_name}')
        self.main_canvas.refresh()

      self.position_setting(label, [0, 0, 2], self.sticky)
      entry.grid_forget()


    # endregion: Button Func

    # (2) Widgets Setting
    # region: Frame Setting
    frame = tk.Frame(
      self.annotation_process_panel,
      bg=self.annotation_process_panel['background'],
      name=f'frame:label-{channel_name}')

    # endregion: Frame Setting

    # region: Label Setting
    label_layer = tk.Label(
      frame, text=channel_name, width=self.width * 2,
      bg=self.annotation_process_panel['background'],
      name='label:layer dynamic')

    # endregion: Label Setting

    # region: Entry Setting
    entry = tk.Entry(frame, name='entry:layer dynamic')

    double_click_callback = partial(
      on_double_click, label=label_layer, entry=entry)
    enter_callback = partial(
      on_enter, label=label_layer, entry=entry, channel_name=channel_name)

    label_layer.bind("<Double-Button-1>", double_click_callback)
    entry.bind("<Return>", enter_callback)
    # self.master.bind("<Button-1>", on_window_click)

    # endregion: Entry Setting

    # region: Button Setting
    button_show = tk.Button(
      frame, text='Show', width=self.width, height=self.height,
      command=lambda n=channel_name: button_show_label(n),
      name='button:show dynamic')

    button_delete = tk.Button(
      frame, text='Delete', width=self.width, height=self.height,
      command=lambda n=channel_name: button_delete_label(n),
      name='button:delete dynamic')

    # endregion: Button Setting

    # (3) Position Setting
    # region: Position Setting in Frame
    self.position_setting(label_layer, [0, 0, 2], self.sticky)
    self.position_setting(button_show, [0, 2, 1], self.sticky)
    self.position_setting(button_delete, [0, 3, 1], self.sticky)

    # endregion: Position Setting in Frame

    # region: Position Setting of Frame
    self.unit_position[self.annotation_process_panel.winfo_name()].update({
      frame.winfo_name(): [row, 0, 1]
    })

    # endregion: Position Setting of Frame


  def display_in_status_box(self, text):
    max_line = 20

    textbox = self.status_panel.children['frame:textbox'].children['text:status']
    textbox.config(state=tk.NORMAL)
    textbox.insert(tk.END, text + "\n")
    textbox.config(state=tk.DISABLED)
    # Check that the number of lines in the text box is more than 10,
    # and if so, delete the oldest line
    lines = textbox.get(1.0, tk.END).split('\n')
    if len(lines) > max_line:
      textbox.config(state=tk.NORMAL)
      textbox.delete(1.0, 2.0)
      textbox.config(state=tk.DISABLED)
