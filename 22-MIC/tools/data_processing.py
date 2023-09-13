import random
import SimpleITK as sitk
import numpy as np

from PIL import Image
from xomics import MedicalImage



# Data Augmentation
def add_gaussian_noise(image: np.ndarray, mean=0, std=1):
  assert len(image.shape) in [2, 3]

  noise = np.random.normal(mean, std, image.shape)
  return image + noise


def image_rotation(image, angle):
  assert len(image.shape) in [2, 3]
  assert angle in [0, 90, 180, 270]
  for _ in range(3):
    if angle <= 0: break
    image = [np.rot90(img) for img in image]
    angle = angle - 90

  return np.array(image)


def image_flip(image, axes):
  assert len(image.shape) in [2, 3]
  return np.flip(image, axis=axes)


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
# Data Preprocessing
def find_max_dimensions(features):
  max_shape = [0, 0, 0]

  for feature in features:
    shape = feature.shape
    for i in range(3):
      max_shape[i] = max(max_shape[i], shape[i])

  return max_shape


def find_nonzero_bounds(array):
  '''
  min_indice = [min_z, min_x, min_y]
  max_indice = [max_z, max_x, max_y]
  '''
  indices = np.nonzero(array)
  min_indices = [np.min(indices[x]) for x in range(3)]
  max_indices = [np.max(indices[x]) for x in range(3)]

  return max_indices, min_indices



if __name__ == "__main__":
  file = r'../data/test_image.jpg'
  image = np.array(Image.open(file))
  images = np.array([image, image, image])
  rotated_images = image_rotation(images, 180)

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1, 4)
  ax[0].imshow(image, cmap='gray')
  ax[1].imshow(rotated_images[0], cmap='gray')
  ax[2].imshow(rotated_images[1], cmap='gray')
  ax[3].imshow(rotated_images[2], cmap='gray')

  plt.show()