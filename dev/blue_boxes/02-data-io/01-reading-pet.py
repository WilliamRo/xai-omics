from roma import finder, console
from xomics.gui.dr_gordon import DrGordon, SliceView, Plotter
from xomics import MedicalImage

import numpy as np



data_dir = r'../../../data/01-ULD-RAW/Subject_1-6/01122021_1_20211201_164050/Full_dose'
im_paths = finder.walk(data_dir, pattern='*.IMA')
n_slices = len(im_paths)

# (1) Read data using pydicom
import pydicom

dicom_slices = []
for i, fp in enumerate(im_paths):
  console.print_progress(i, n_slices)
  s = pydicom.dcmread(fp)
  dicom_slices.append(s)
dicom_slices = sorted(dicom_slices, key=lambda s: s.InstanceNumber,
                      reverse=True)
console.show_status(f'Successfully read {n_slices} slices using pydicom.')

dicom_data = np.stack([s.pixel_array for s in dicom_slices], axis=0)
dicom_data = dicom_data.astype(np.float64)



# (2) Read data using simpleITK
import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames((data_dir)))
sitk_image = reader.Execute()
sitk_data = sitk.GetArrayFromImage(sitk_image)



# (*) Visualize data using Dr. Gordon
def visualize_data():
  mi = MedicalImage('Subject-?', images={
    'dicom_data': dicom_data,
    'sikt_data': sitk_data,
  })

  dg = DrGordon([mi])
  dg.slice_view.set('vmin', auto_refresh=False)
  dg.slice_view.set('vmax', auto_refresh=False)

  dg.show()



# (*) Analyze slice distribution
def analyze_slice_energy_distribution():
  import matplotlib.pyplot as plt

  console.show_info(
    f'dicom data range: {np.min(dicom_data)} - {np.max(dicom_data)}')
  console.show_info(
    f'sitk data range: {np.min(sitk_data)} - {np.max(sitk_data)}')

  dicom_Es = np.sum(dicom_data, axis=(1, 2))
  dicom_E = np.sum(dicom_Es)
  dicom_dist = dicom_Es / dicom_E

  sitk_Es = np.sum(sitk_data, axis=(1, 2))
  sitk_E = np.sum(sitk_Es)
  sitk_dist = sitk_Es / sitk_E

  plt.plot(dicom_dist)
  plt.plot(sitk_dist, ':')
  plt.legend(['pydicom', 'SimpleITK'])
  plt.show()


analyze_slice_energy_distribution()


