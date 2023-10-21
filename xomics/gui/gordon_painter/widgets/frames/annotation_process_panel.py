from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase
from functools import partial

import tkinter as tk



class AnnotationProcessPanel(FrameBase):
  def __init__(
      self, master, bg='lightgray', name='frame:annotation_process_panel'):
    super().__init__(master=master, bg=bg, name=name)


  def _init_panel(self):
    # (1) Button Function Setting
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
      self.master.status_panel.display_in_status_box(
        f'Button Painter set to {self.slice_view.get("painter")}')


    # (2) Widgets Setting
    # region: Parameter Setting
    percentile_step = self.main_canvas.configs['percentile_step']
    # endregion: Parameter Setting

    # region: Frame Setting
    frame_painter = tk.Frame(
      self, bg=self.bg ,name='frame:painter')

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
      width=self.width * 2, bg=self.bg, name='label:percentile')

    label_mask_prompt = tk.Label(
      frame_painter, text='All labels are below: ', width=self.width * 4,
      bg=self.bg, name='label:mask prompt')

    # endregion: Label Setting

    # (3) Position Setting in Frame
    # region: Position Setting of frame_painter
    self.position_setting(button_sub_percentile, [0, 0, 1], self.sticky)
    self.position_setting(label_percentile, [0, 1, 2], self.sticky)
    self.position_setting(button_add_percentile, [0, 3, 1], self.sticky)
    self.position_setting(button_painter, [1, 0, 4], self.sticky)
    self.position_setting(label_mask_prompt, [2, 0, 4], self.sticky)

    # endregion: Position Setting of frame_painter

    # (4) Dynamic widgets Setting
    # region: Dynamic Frame Setting
    for i, layer in enumerate(self.main_canvas.channels):
      channel_name = list(
        self.slice_view.selected_medical_image.labels.keys())[layer]

      self.create_dynamic_label_frame(channel_name, i + 2)
    # endregion: Dynamic Frame Setting


  def create_dynamic_label_frame(self, channel_name, row):
    # (1) Button Function Setting
    # region: Button Func
    def button_show_label(label_name):
      if label_name in self.slice_view.annotations_to_show:
        self.slice_view.annotations_to_show.remove(label_name)
        self.master.status_panel.display_in_status_box(
          f'{label_name} hided.')
      else:
        self.slice_view.annotations_to_show.append(label_name)
        self.master.status_panel.display_in_status_box(
          f'{label_name} showed.')

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
        label_data = self.slice_view.selected_medical_image.labels[
          channel_name]
        self.slice_view.selected_medical_image.labels[
          new_channel_name] = label_data
        del self.slice_view.selected_medical_image.labels[channel_name]
        self.master.status_panel.display_in_status_box(
          f'rename label-{channel_name} to label-{new_channel_name}')
        self.main_canvas.refresh()

      self.position_setting(label, [0, 0, 2], self.sticky)
      entry.grid_forget()

    # endregion: Button Func

    # (2) Widgets Setting
    # region: Frame Setting
    frame = tk.Frame(self, bg=self.bg, name=f'frame:label-{channel_name}')

    # endregion: Frame Setting

    # region: Label Setting
    label_layer = tk.Label(
      frame, text=channel_name, width=self.width * 2, bg=self.bg,
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

    # (3) Position Setting in Frame
    # region: Position Setting of frame
    self.position_setting(label_layer, [0, 0, 2], self.sticky)
    self.position_setting(button_show, [0, 2, 1], self.sticky)
    self.position_setting(button_delete, [0, 3, 1], self.sticky)

    # endregion: Position Setting in Frame

    # region: Position Setting of Frame
    self.unit_position[self.winfo_name()].update({
      frame.winfo_name(): [row, 0, 1]
    })

    # endregion: Position Setting of Frame



if __name__ == '__main__':
  pass