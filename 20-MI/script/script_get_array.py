import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import pickle

from tqdm import tqdm



def dilation_and_fill(img):
  new_img = np.zeros_like(img, dtype=np.uint8)

  # dilation
  kernel = np.ones((2, 2), np.uint8)
  dilated_img = cv2.dilate(img, kernel, iterations=1)

  # fill
  ret, thresh = cv2.threshold(np.expand_dims(dilated_img, axis=-1), 127,
                              255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(contours)):
    cv2.drawContours(new_img, contours, i, 255, -1)

  new_img[new_img != 0] = 1

  return new_img


if __name__ == '__main__':
  structure_file = '../data/dataset/ANON10066/RS.1.2.246.352.205.5720313517654988668.11384827600675937442.dcm'
  ct_dir = '../data/dataset/ANON10066'
  save_file = '../data/npy_data/ANON10066.npy'

  structure_dcm = pydicom.dcmread(structure_file)

  structure_sequence = structure_dcm.StructureSetROISequence
  contour_sequence = structure_dcm.ROIContourSequence

  # get CTV_NUMBER
  CTV_NUMBER = next(
    (sequence.ROINumber for sequence in structure_sequence if
     sequence.ROIName == 'CTV'), None)
  assert CTV_NUMBER is not None

  CTV_contour_seq = next(
    (sequence.ContourSequence for sequence in contour_sequence if
     sequence.ReferencedROINumber == CTV_NUMBER), None)
  assert CTV_contour_seq is not None

  ctv_array_dict = {}
  for i in CTV_contour_seq:
    ctv_array_dict[
      str(i.ContourImageSequence[0].ReferencedSOPInstanceUID)] = np.array(
      i.ContourData, dtype=np.float32).reshape(i.NumberOfContourPoints, 3)

  my_dict = {}
  ct_names = os.listdir(os.path.join(ct_dir))
  ct_names = [name for name in ct_names if 'CT' in name]

  for name in ct_names:
    ct_file = os.path.join(ct_dir, name)
    ct_dcm = pydicom.dcmread(ct_file)
    ct_dict = {'ct_array': np.array(ct_dcm.pixel_array),
               'pixel_spacing': np.array(ct_dcm.PixelSpacing),
               'slice_location': float(ct_dcm.SliceLocation),
               'slice_thickness': float(ct_dcm.SliceThickness),
               'window_center': float(ct_dcm.WindowCenter),
               'window_width': float(ct_dcm.WindowWidth),
               'instance_number': int(ct_dcm.InstanceNumber),
               'image_position': np.array(ct_dcm.ImagePositionPatient),
               'ctv_array_original': ctv_array_dict[str(ct_dcm.SOPInstanceUID)]
               if str(ct_dcm.SOPInstanceUID) in ctv_array_dict else None}

    if ct_dict['ctv_array_original'] is not None:
      assert ct_dict['slice_location'] == ct_dict['ctv_array_original'][0][2]
    my_dict[name] = ct_dict

  # transform ctv_array
  for key in my_dict:
    ctv_array_original = my_dict[key]['ctv_array_original']
    if ctv_array_original is None:
      my_dict[key]['ctv_array'] = np.zeros_like(my_dict[key]['ct_array'], dtype=np.uint8)
    else:
      image_position = my_dict[key]['image_position']
      pixel_spacing = my_dict[key]['pixel_spacing']

      x_array = np.array([(array[0] - image_position[0]) / pixel_spacing[0] for array in ctv_array_original], dtype=int)
      y_array = np.array([(array[1] - image_position[1]) / pixel_spacing[1] for array in ctv_array_original], dtype=int)
      coords = np.transpose(np.array([y_array, x_array]), axes=(1, 0))
      ctv_coords = np.zeros_like(my_dict[key]['ct_array'], dtype=np.uint8)
      for coord in coords:
        ctv_coords[coord[0], coord[1]] = 255
      my_dict[key]['ctv_array'] = dilation_and_fill(ctv_coords)

  for name in my_dict:
    for key in list(my_dict[name].keys()):
      b = my_dict[name][key]
      if key not in ('ct_array', 'ctv_array'):
        del my_dict[name][key]


  np.save(save_file, my_dict)
  print(f'{save_file} is saved successfully!!!')