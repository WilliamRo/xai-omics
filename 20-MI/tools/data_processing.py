from tqdm import tqdm
from xomics.objects import MedicalImage
from xomics.gui.dr_gordon import DrGordon

import numpy as np
import math
import os



# input_size = (?, 512, 512, ?)
def data_separation(input_matrix, cut_size):
  assert len(input_matrix.shape) == 4
  # get pad size
  original_size_x, original_size_y, pad_size = (
    input_matrix.shape[1], input_matrix.shape[2], cut_size // 2)
  total_size_x = (math.ceil(original_size_x / cut_size) * cut_size +
                  2 * pad_size)
  total_size_y = (math.ceil(original_size_y / cut_size) * cut_size +
                  2 * pad_size)
  end_pad_size_x = total_size_x - pad_size - original_size_x
  end_pad_size_y = total_size_y - pad_size - original_size_y
  section_size = cut_size + 2 * pad_size

  # pad
  pad_width = ((0, 0), (pad_size, end_pad_size_x),
               (pad_size, end_pad_size_y), (0, 0))

  pad_matrix = np.pad(input_matrix, pad_width=pad_width, mode='constant')

  # cut
  cut_index_x = [i for i in
                 range(0, total_size_x - section_size + 1, cut_size)]
  cut_index_y = [i for i in
                 range(0, total_size_y - section_size + 1, cut_size)]

  result = []
  for i in range(len(cut_index_x)):
    for j in range(len(cut_index_y)):
      result.append(
        pad_matrix[:, cut_index_x[i]:cut_index_x[i] + section_size,
        cut_index_y[j]:cut_index_y[j] + section_size, :])

  result = np.array(result)

  assert len(result.shape) == 5

  my_dict = {'original_size': input_matrix.shape,
             'cut_index_x': cut_index_x,
             'cut_index_y': cut_index_y,
             'pad_size': pad_size,
             'cut_size': cut_size,
             'end_pad_size_x': end_pad_size_x,
             'end_pad_size_y': end_pad_size_y}

  return result, my_dict


def data_joint(separated_matrix, my_dict):
  final_image = np.zeros(my_dict['original_size'])
  cut_index_x, cut_index_y = my_dict['cut_index_x'], my_dict['cut_index_y']
  pad_size = my_dict['pad_size']
  cut_size = my_dict['cut_size']
  end_pad_size_x, end_pad_size_y = (my_dict['end_pad_size_x'],
                                    my_dict['end_pad_size_y'])

  for i in range(len(cut_index_x)):
    for j in range(len(cut_index_y)):
      add_size_x = (cut_size if cut_index_x[i] != cut_index_x[-1] else
                    cut_size - (end_pad_size_x - pad_size))
      add_size_y = (cut_size if cut_index_y[j] != cut_index_y[-1] else
                    cut_size - (end_pad_size_y - pad_size))
      num = j + i * len(cut_index_y)
      final_image[:, cut_index_x[i]:cut_index_x[i] + add_size_x,
      cut_index_y[j]:cut_index_y[j] + add_size_y, :] = \
        separated_matrix[num, :, pad_size:pad_size + add_size_x,
        pad_size:pad_size + add_size_y, :]

  return final_image


def save_as_npy(file_path, input_list):
  '''
  The input is a list
  example: list = [[1,2,3], [4, 5], [6, 7, 8, 9]]
  '''
  array = np.array(input_list, dtype=object)
  np.save(file_path, array)


def load_npy_file(file_path):
  return np.load(file_path, allow_pickle=True).tolist()


def get_segmentation_data(input_data: np.ndarray, type=None):
  '''
  input: datas in numpy
         datas shape: (slice, length, width) (97, 512, 512)
  output: liver datas or tumor datas
          datas shape: (slice, length, width) (97, 512, 512)
  '''
  if type == 'liver':
    type = 1
  elif type == 'tumor':
    type = 2
  else:
    raise TypeError("The 'type' parameter is incorrect!")
  index = np.where(input_data == type)
  output_data = np.zeros_like(input_data)
  output_data[index] = type

  return output_data


def mi_add_gaussian_noise(mi: MedicalImage, mean=0, std=1):
  noise = np.random.normal(mean, std, mi.shape)
  for key in mi.images.keys():
    image = mi.images[key]
    mi.images[key] = image + noise


def mi_rotation(mi: MedicalImage, angle):
  assert angle in [0, 90, 180, 270]

  # images
  for key in mi.images.keys():
    ang = angle
    image = mi.images[key]
    for _ in range(3):
      if ang <= 0: break
      image = [np.rot90(img) for img in image]
      ang = ang - 90

    mi.images[key] = np.array(image)

  # labels
  for key in mi.labels.keys():
    ang = angle
    label = mi.labels[key]
    for _ in range(3):
      label = [np.rot90(lab) for lab in label]
      ang = ang - 90
      if ang <= 0: break

    mi.labels[key] = np.array(label)


def mi_flip(mi: MedicalImage, axes):
  assert axes in [0, 1, 2]

  # images
  for key in mi.images.keys():
    image = mi.images[key]
    mi.images[key] = np.flip(image, axis=axes)

  # labels
  for key in mi.labels.keys():
    label = mi.labels[key]
    mi.labels[key] = np.flip(label, axis=axes)



if __name__ == "__main__":
  image_dir = r'../../data/02-PET-CT-Y1/mi'
  image_names = os.listdir(image_dir)

  for i, name in enumerate(image_names):
    image_path = os.path.join(image_dir, name)
    mi: MedicalImage = MedicalImage.load(image_path)

    mi.window('ct', -200, 300)
    mi.normalization(['ct', 'pet'])

    mi_rotation(mi, 180)
    mi_flip(mi, 2)
    mi_add_gaussian_noise(mi, 0, 1)
    if '0011' in name:
      print()
    print(i)
    print(name)

  # Visualization
  # dg = DrGordon([mi])
  # dg.slice_view.set('vmin', auto_refresh=False)
  # dg.slice_view.set('vmax', auto_refresh=False)
  # dg.show()
  # print()
