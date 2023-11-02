from xomics import MedicalImage
from tkinter import messagebox
from xomics.data_io.utils.preprocess import calc_SUV

import tkinter as tk
import numpy as np
import nibabel as nib
import pydicom
import os
import re



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
  def open_dicom_file(self, file_path=None):
    def pet_image_precess(data, dcm):
      # Calculate the suv
      ST = dcm.SeriesTime
      AT = dcm.AcquisitionTime
      PW = dcm.PatientWeight
      RIS = dcm.RadiopharmaceuticalInformationSequence[0]
      RST = str(RIS['RadiopharmaceuticalStartTime'].value)
      RTD = str(RIS['RadionuclideTotalDose'].value)
      RHL = str(RIS['RadionuclideHalfLife'].value)
      RS = dcm.RescaleSlope
      RI = dcm.RescaleIntercept
      dcm_tag = {
        'ST': ST,
        'AT': AT,
        'PW': PW,
        'RST': RST,
        'RTD': RTD,
        'RHL': RHL,
        'RS': RS,
        'RI': RI
      }
      return calc_SUV(data, tags=dcm_tag, norm=False)

    def ct_image_process(data, dcm):
      RS = dcm.RescaleSlope
      RI = dcm.RescaleIntercept

      return data * RS + RI

    def read_dcm_file(dcm_dir):
      dcm_file = os.listdir(dcm_dir)

      dcms = [pydicom.dcmread(os.path.join(dcm_dir, file))
              for file in dcm_file]
      dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
      data = np.array([d.pixel_array for d in dcms], dtype=np.uint16)

      dcm = dcms[0]
      pid = str(dcm.PatientName).lower().replace(' ', '')
      imagetype = str(dcm.SeriesDescription).split(' ')[0].lower()

      if imagetype == 'ct':
        data = ct_image_process(data, dcm)
      elif imagetype == 'pet':
        data = pet_image_precess(data, dcm)
      else:
        data = None

      return data, pid, imagetype


    if not file_path:
      file_path = tk.filedialog.askopenfilename(title="选择DICOM文件")
      if not file_path: return

    dcm_dir = os.path.dirname(file_path)
    data, pid, image_type = read_dcm_file(dcm_dir)
    if data is None: return
    mi = self.slice_view.selected_medical_image
    if mi and pid == mi.key:
      if mi.shape == data.shape:
        mi.images[image_type] = data
      else:
        messagebox.showerror(
          'Error', "The shape of input does not match the image's shape")
    else:
      mi = MedicalImage(images={image_type: data}, key=pid)
      mi.normalization([image_type])
      mi_list = [mi]
      self.main_canvas.set_data(mi_list)

    self.main_canvas.refresh()


  def open_nifti_file(self, file_paths=None):
    def show_select_top():
      def select(choices):
        choice.set(choices)
        top_select.destroy()

      # (1) Toplevel Setting
      top_select = tk.Toplevel(self)
      top_select.title('Image or Segmentation')

      # (2) Button Setting
      button_image = tk.Button(
        top_select, text='Load as Image', width=self.width * 2,
        command=lambda c='image': select(c))
      button_segmentation = tk.Button(
        top_select, text='Load as Segmentation', width=self.width * 2,
        command=lambda c='segmentation': select(c))

      # (3) Position Setting in Toplevel
      self.position_setting(button_image, [0, 0, 2], self.sticky)
      self.position_setting(button_segmentation, [0, 2, 2], self.sticky)

      import time
      time.sleep(0.1)
      # (4) Position Setting of Toplevel
      old_geometry = re.findall(
        r'\d+', self.main_canvas.master.winfo_geometry())
      width, height, width_inc, height_inc = [int(n) for n in old_geometry]
      new_width, new_height = 300, 30
      new_width_inc = width_inc + width // 2 - new_width // 2
      new_height_inc = height_inc + height // 2 - new_height //2
      top_select.geometry(
        '{}x{}+{}+{}'.format(new_width, new_height,
                             new_width_inc, new_height_inc))

      self.wait_window(top_select)

    if not file_paths:
      file_paths = tk.filedialog.askopenfilename(title="选择NIfTI文件")
      if not file_paths: return
    choice = tk.StringVar()
    choice.set(None)
    show_select_top()

    data = np.transpose(
      np.array(nib.load(file_paths).get_fdata()), axes=(2, 1, 0))

    if choice.get() == 'image':
      mi_list = [MedicalImage(images={'unknown': data}, key='unknown')]
      self.main_canvas.set_data(mi_list)
      self.main_canvas.refresh()
    elif choice.get() == 'segmentation':
      mi = self.slice_view.selected_medical_image
      if not mi:
        messagebox.showerror(
          'Error', 'No segmentation can be added without image')
      elif mi.shape != data.shape:
        messagebox.showerror(
          'Error', "The shape of segmentation does not match the image's shape")
      else:
        label_num = [int(l.split('-')[-1])
                     for l in mi.labels.keys() if 'label-' in l]
        label_num = max(label_num) + 1 if label_num else 0
        mi.labels[f'label-{label_num}'] = data
        self.main_canvas.refresh()


  def open_mi_file(self, file_paths=None):
    if not file_paths:
      file_paths = tk.filedialog.askopenfilenames(
        title="选择MI文件", filetypes=[('MI文件', '*.mi')])
      if not file_paths: return

    mi_list = [MedicalImage.load(f) for f in file_paths]
    self.main_canvas.set_data(mi_list)
    self.main_canvas.refresh()


  # endregion: Menu Func

  def position_setting(self, element, distribution, sticky):
    row, column, columnspan = distribution
    element.grid(
      row=row, column=column, columnspan=columnspan, sticky=sticky)


if __name__ == '__main__':
  pass