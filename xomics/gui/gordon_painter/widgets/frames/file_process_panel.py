from xomics.gui.gordon_painter.widgets.frames.frame_base import FrameBase

import tkinter as tk



class FileProcessPanel(FrameBase):
  def __init__(self, master, bg='lightgray', name='frame:file_process_panel'):
    super().__init__(master=master, bg=bg, name=name)


  def _init_panel(self):
    # (1) Button Function Setting
    def mi_save():
      file_type = [('MI文件', '*.mi')]
      mi = self.slice_view.selected_medical_image
      initial_file = mi.key + '.mi'

      file_path = tk.filedialog.asksaveasfilename(
        filetypes=file_type, initialfile=initial_file)
      if not file_path: return
      mi.save(file_path)
      self.master.status_panel.display_in_status_box(
        f'mi files saved successfully in {file_path}')


    def nii_save():
      dir_path = tk.filedialog.askdirectory()
      if not dir_path: return
      mi = self.slice_view.selected_medical_image
      mi.save_as_nii(dir_path)

      self.master.status_panel.display_in_status_box(
        f'nii saved successfully in {dir_path}')


    # (2) Widgets Setting
    # region: Frame Setting
    frame_mi_save = tk.Frame(
      self, bg=self['background'], name='frame:mi save')
    frame_nii_save = tk.Frame(
      self, bg=self['background'], name='frame:nii save')

    # endregion: Frame Setting

    # region: Button Setting
    button_mi_save = tk.Button(
      frame_mi_save, text="Save as mi", width=self.width * 2,
      height=self.height, command=lambda: mi_save(), name='button:mi save')

    button_nii_save = tk.Button(
      frame_nii_save, text="Save as nii", width=self.width * 2,
      height=self.height, command=lambda: nii_save(), name='button:nii save')

    # endregion: Button Setting

    # region: Label Setting
    label_mi_save = tk.Label(
      frame_mi_save, text='MI: ', width=self.width * 2,
      bg=self['background'], name='label:mi save')
    label_nii_save = tk.Label(
      frame_nii_save, text='NIfTI: ', width=self.width * 2,
      bg=self['background'], name='label:nii save')

    # endregion: Label Setting

    # (3) Position Setting in Frame
    # region: Position Setting of frame_mi_save
    self.position_setting(label_mi_save, [0, 0, 2], self.sticky)
    self.position_setting(button_mi_save, [0, 2, 2], self.sticky)

    # endregion: Position Setting of frame_mi_save

    # region: Position Setting of frame_nii_save
    self.position_setting(label_nii_save, [0, 0, 2], self.sticky)
    self.position_setting(button_nii_save, [0, 2, 2], self.sticky)

    # endregion: Position Setting of frame_nii_save



if __name__ == '__main__':
  pass