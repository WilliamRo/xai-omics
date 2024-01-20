import numpy as np
from scipy.spatial.distance import cdist



def hausdorff_distance(coord1, coord2, percentile, physical_distance):
  distance_matrix = cdist(coord1 * physical_distance,
                          coord2 * physical_distance)
  a2b = np.percentile(np.sort(np.min(distance_matrix, axis=1)), percentile)
  b2a = np.percentile(np.sort(np.min(distance_matrix, axis=0)), percentile)
  return max(a2b, b2a)


def dice_accuarcy(ground_truth, prediction):
  assert ground_truth.shape == prediction.shape
  smooth = 1.0

  intersection = np.sum(ground_truth * prediction)
  acc = ((2.0 * intersection + smooth) /
         (np.sum(ground_truth) + np.sum(prediction) + smooth))

  return acc



if __name__ == '__main__':
  pass