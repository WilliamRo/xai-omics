from tqdm import tqdm
from collections import OrderedDict

import tkinter as tk
import os



class ToolPanel(tk.Frame):
  def __init__(self, master, bg='lightgray'):
    super().__init__(master=master, bg=bg)

    # Parameter Setting
    self.unit_position = OrderedDict()

    # Create panels
    self.status_panel = tk.Frame(
      self, bg='gray', name='frame:status panel')
    self.image_switch_panel = tk.Frame(
      self, bg='lightgray', name='frame:image switch panel')
    self.annotation_process_panel = tk.Frame(
      self, bg='gray', name='frame:annotation process panel')

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

  # @property
  # def unit_position(self):
  #   position_dict = {}
  #   for p in self.children.values():
  #     position_dict[p.winfo_name()] = OrderedDict()
  #
  #   return position_dict

  @property
  def slice_view(self):
    return self.main_canvas.slice_view

  @property
  def percentile_step(self):
    return self.main_canvas.configs['percentile_step']

  # endregion: Properties

  # region: Init Func
  def _init_tool_panel(self):
    self._init_layout()
    self._init_position_dict()
    self._create_status_panel()
    self._create_image_switch_panel()
    self._create_annotation_process_panel()


  def _init_layout(self):
    self.status_panel.pack(side='top', fill='both', expand=True)
    self.image_switch_panel.pack(side='top', fill='both', expand=True)
    self.annotation_process_panel.pack(
      side='top', fill='both', expand=True)


  def _init_position_dict(self):
    for p in self.children.values():
      self.unit_position.update({
        p.winfo_name(): OrderedDict()
      })

  # endregion: Init Func

  def _create_status_panel(self):
    pass


  def _create_image_switch_panel(self):
    width = self.main_canvas.style_sheet['unit_width']
    height = self.main_canvas.style_sheet['unit_height']
    sticky = self.main_canvas.style_sheet['sticky']

    # Frame Setting
    frame_patient = tk.Frame(
      self.image_switch_panel, bg='lightgray', name='frame:patient')
    frame_channel = tk.Frame(
      self.image_switch_panel, bg='lightgray', name='frame:channel')

    # Button Setting
    button_next_patient = tk.Button(
      frame_patient, text="Next", width=width, height=height,
      command=lambda: self.main_canvas._set_patient_cursor(1),
      name='button:next patient')

    button_pre_patient = tk.Button(
      frame_patient, text="Pre", width=width, height=height,
      command=lambda: self.main_canvas._set_patient_cursor(-1),
      name='button:pre patient')

    button_next_channel = tk.Button(
      frame_channel, text="Next", width=width, height=height,
      command=lambda: self.main_canvas.set_cursor(
        self.main_canvas.Keys.LAYERS, 1, refresh=True),
      name='button:next channel')

    button_pre_channel = tk.Button(
      frame_channel, text="Pre", width=width, height=height,
      command=lambda: self.main_canvas.set_cursor(
        self.main_canvas.Keys.LAYERS, -1,refresh=True),
      name='button:pre channel')

    # Text Setting
    self.text_var['patient'] = tk.StringVar()
    text_patient = tk.Label(
      frame_patient, textvariable=self.text_var['patient'],
      width=width * 2, bg='lightgray', name='label:patient')

    self.text_var['channel'] = tk.StringVar()
    text_channel = tk.Label(
      frame_channel, textvariable=self.text_var['channel'],
      width=2 * width, bg='lightgray', name='label:channel')

    # Position Setting in Frame
    self.position_setting(text_patient, [0, 0, 2], sticky)
    self.position_setting(button_next_patient, [0, 2, 1], sticky)
    self.position_setting(button_pre_patient, [0, 3, 1], sticky)

    self.position_setting(text_channel, [0, 0, 2], sticky)
    self.position_setting(button_next_channel, [0, 2, 1], sticky)
    self.position_setting(button_pre_channel, [0, 3, 1], sticky)

    # Position Setting out of Frame
    self.unit_position[self.image_switch_panel.winfo_name()].update({
      frame_patient.winfo_name(): [0, 0, 1],
      frame_channel.winfo_name(): [1, 0, 1]
    })


  def _create_annotation_process_panel(self):
    # Button Functions
    def button_set_percentile(step):
      if self.slice_view.percentile and self.slice_view.click_address:
        self.slice_view.percentile = self.slice_view.percentile + step

        # restricted range
        self.slice_view.percentile = min(
          100.00, self.slice_view.percentile)
        self.slice_view.percentile = max(0.00, self.slice_view.percentile)

        self.slice_view.adjust_mask(self.slice_view.percentile)
        self.main_canvas.refresh()

    # Parameter Setting
    width = self.main_canvas.style_sheet['unit_width']
    height = self.main_canvas.style_sheet['unit_height']
    sticky = self.main_canvas.style_sheet['sticky']
    percentile_step = self.percentile_step

    # Frame Setting
    frame_painter = tk.Frame(
      self.annotation_process_panel, bg='lightgray', name='frame:painter')

    # Button Setting
    button_painter = tk.Button(
      frame_painter, text="Painter", width=2 * width, height=height,
      command=lambda: self.slice_view.flip('painter'),
      name='button:painter')

    button_add_percentile = tk.Button(
      frame_painter, text="Add", width=width, height=height,
      command=lambda s=percentile_step: button_set_percentile(s),
      name='button:add percentile')

    button_sub_percentile = tk.Button(
      frame_painter, text="Sub", width=width, height=height,
      command=lambda s=percentile_step: button_set_percentile(-s),
      name='button:sub percentile')

    # Text Setting
    self.text_var['percentile'] = tk.StringVar()
    text_percentile = tk.Label(
      frame_painter, textvariable=self.text_var['percentile'],
      width=4 * width, bg='gray', name='label:percentile')

    text_mask_prompt = tk.Label(
      frame_painter, text='All labels are below: ', width=4 * width,
      bg='gray', name='label:mask prompt')

    # Position Setting in Frame
    self.position_setting(text_percentile, [0, 0, 4], sticky)
    self.position_setting(button_painter, [1, 0, 2], sticky)
    self.position_setting(button_add_percentile, [1, 2, 1], sticky)
    self.position_setting(button_sub_percentile, [1, 3, 1], sticky)
    self.position_setting(text_mask_prompt, [2, 0, 4], sticky)

    # Position Setting out of Frame
    self.unit_position[self.annotation_process_panel.winfo_name()].update({
      frame_painter.winfo_name(): [0, 0, 1]
    })

    # Dynamic Button Setting
    for i, layer in enumerate(self.main_canvas.channels):
      channel_name = list(
        self.slice_view.selected_medical_image.labels.keys())[layer]

      self.create_dynamic_label_frame(channel_name, i + 1)


  def position_setting(self, element, distribution, sticky):
    row, column, columnspan = distribution
    element.grid(
      row=row, column=column, columnspan=columnspan, sticky=sticky)


  def create_dynamic_label_frame(self, channel_name, row):
    def button_show_label(label_name):
      if label_name in self.slice_view.annotations_to_show:
        self.slice_view.annotations_to_show.remove(label_name)
      else:
        self.slice_view.annotations_to_show.append(label_name)

      self.main_canvas.refresh()


    def button_delete_label(label_name):
      del self.slice_view.selected_medical_image.labels[label_name]
      self.main_canvas.refresh()

    # Parameter Setting
    width = self.main_canvas.style_sheet['unit_width']
    height = self.main_canvas.style_sheet['unit_height']
    sticky = self.main_canvas.style_sheet['sticky']

    # Frame Setting
    frame = tk.Frame(
      self.annotation_process_panel, bg='gray',
      name=f'frame:label-{channel_name}')

    # Text Setting
    text_layer = tk.Label(
      frame, text=channel_name, width=2 * width, bg='gray',
      name=f'label:layer {channel_name}')

    # Button Setting
    button_show = tk.Button(
      frame, text='Show', width=width, height=height,
      command=lambda n=channel_name: button_show_label(n),
      name=f'button:show {channel_name}')

    button_delete = tk.Button(
      frame, text='Delete', width=width, height=height,
      command=lambda n=channel_name: button_delete_label(n),
      name=f'button:delete {channel_name}')

    self.position_setting(text_layer, [0, 0, 2], sticky)
    self.position_setting(button_show, [0, 2, 1], sticky)
    self.position_setting(button_delete, [0, 3, 1], sticky)

    self.unit_position[self.annotation_process_panel.winfo_name()].update({
      frame.winfo_name(): [row, 0, 1]
    })

