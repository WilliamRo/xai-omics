from xomics import MedicalImage
from tqdm import tqdm

import tkinter as tk
import os



class MenuBar(tk.Menu):
  def __init__(self, master, bg='lightgray'):
    super().__init__(master=master, bg=bg)

    self._init_menu_bar()


  # region: Properties
  @property
  def main_canvas(self):
    for key in self.master.children.keys():
      if 'gordonpainter' in key:
        return self.master.children[key]

  # endregion: Properties

  # region: Init Func
  def _init_menu_bar(self):
    # create File menu
    file_menu = tk.Menu(self, tearoff=0)
    self.add_cascade(label="File", menu=file_menu)

    # add the menu items to the file menu
    file_menu.add_command(label="Open DICOM file",
                          command=self.open_dicom_file)
    file_menu.add_command(label="Open NIfTI file",
                          command=self.open_nifti_file)
    file_menu.add_command(label="Open MI file",
                          command=self.open_mi_file)
    file_menu.add_separator()

    file_menu.add_command(label="Exit", command=self.master.quit)

  # endregion: Init Func

  # region: Menu Func
  def open_dicom_file(self, file_paths=None):
    if not file_paths:
      dir_path = tk.filedialog.askdirectory(title="选择DICOM文件夹")
      if not dir_path: return
    # TODO --------------------


  def open_nifti_file(self, file_paths=None):
    if not file_paths:
      file_paths = tk.filedialog.askopenfilenames(title="选择NIfTI文件")
      if not file_paths: return
    # TODO --------------------


  def open_mi_file(self, file_paths=None):
    if not file_paths:
      file_paths = tk.filedialog.askopenfilenames(
        title="选择MI文件", filetypes=[('MI文件', '*.mi')])
      if not file_paths: return

    mi_list = [MedicalImage.load(f) for f in file_paths]
    self.main_canvas.set_data(mi_list)
    self.main_canvas.refresh()


  # endregion: Menu Func



if __name__ == '__main__':
  pass