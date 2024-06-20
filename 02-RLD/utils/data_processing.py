import numpy as np
from skimage import restoration


# self.features/targets.shape = [N, S, H, W, 2]
def get_center(arr: np.ndarray, size):
  start_x = (arr.shape[-2] - size) // 2
  start_y = (arr.shape[-3] - size) // 2

  return arr[:, :, start_x:start_x + size, start_y:start_y + size]


def normalize(arr: np.array):
  norm = np.linalg.norm(arr, ord=1)
  return arr / norm


def windows_choose(distr: np.ndarray, windows_size):
  assert np.NaN not in distr
  x = np.linspace(0, distr.shape[0] - 1, distr.shape[0])
  result = np.random.choice(x, p=distr)
  result = result - windows_size / 2

  if result < 0: result = 0
  if result > distr.shape[0] - windows_size:
    result = distr.shape[0] - windows_size

  return int(result)


def get_random_window(arr: np.ndarray, windows_size, true_rand=False):
  # todo: true random
  assert len(arr.shape) == len(windows_size) + 1

  arr = arr != 0
  arr = arr[:, ..., 0]
  pos = []

  if true_rand:
    for i, size in enumerate(arr.shape):
      index = np.random.randint(size)
      pos.append(windows_choose_simple(index, windows_size[i], size))
    return pos

  dimension = len(arr.shape)
  sub_arr = []
  for i in range(dimension):
    sub_arr.append(np.any(arr, axis=tuple([j for j in range(dimension) if j != i])))

  dist_list = []
  for s_arr in sub_arr:
    dist_list.append(normalize(s_arr.ravel()))

  for i, dist in enumerate(dist_list):
    pos.append(windows_choose(dist, windows_size[i]))

  return pos


def get_sample(arr: np.ndarray, pos, windows_size):
  assert len(pos) == len(windows_size)

  for i, p, s in zip(range(len(pos)), pos, windows_size):
    arr = np.take(arr, range(p, p + s), axis=i)
  return arr


def gen_windows(arr1: np.ndarray, arr2: np.ndarray, batch_size,
                windows_size, **kwargs):
  features = []
  targets = []
  index = np.random.choice(range(arr1.shape[0]), batch_size)
  arr1, arr2 = arr1[index], arr2[index]

  for i in range(batch_size):
    pos = get_random_window(arr1[i], windows_size, **kwargs)
    features.append(get_sample(arr1[i], pos, windows_size))
    targets.append(get_sample(arr2[i], pos, windows_size))
  features = np.stack(features)
  targets = np.stack(targets)

  return features, targets


def nonlocal_mean(image, h=10, patch_size=7, patch_dis=21):
  de_img = restoration.denoise_nl_means(image, h=h,
                                        patch_size=patch_size,
                                        patch_distance=patch_dis)

  return de_img


def windows_choose_simple(pos, windows_size, max_size):
  result = pos - windows_size / 2

  if result < 0: result = 0
  if result > max_size - windows_size:
    result = max_size - windows_size

  return int(result)





if __name__ == '__main__':
  from xomics import MedicalImage
  from xomics.objects.jutils.objects import GeneralMI
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer
  import datetime

  test_gen_win = True

  test = GeneralMI.get_test_sample(r'../../data/02-RLD/rld_data.csv')
  test.process_param['shape'] = [440, 440, 480]
  test.process_param['norm'] = 'PET'

  a = test.images['CT'][0]
  b = test.images['30G'][1]
  if test_gen_win:
    a = np.expand_dims(a, axis=[-1, 0])
    b = np.expand_dims(b, axis=[-1, 0])

    num = 16
    time1 = datetime.datetime.now()
    img_f, img_t = gen_windows(a[0], b[0], num, [128, 128, 128], true_rand=True)
    time2 = datetime.datetime.now()
    print("time:", time2 - time1)
    di_f = {}
    di_t = {}
    for i in range(num):
      di_f[f'feature-{i}'] = img_f[i]
    for i in range(num):
      di_t[f'target-{i}'] = img_t[i]
    mi_f = MedicalImage('feature', di_f)
    mi_t = MedicalImage('target', di_t)
    mis = [mi_f, mi_t]
  else:
    da = nonlocal_mean(a, h=1000, patch_size=3, patch_dis=3)
    print(da.shape)
    mis = [MedicalImage('feature', {'nonlocal': da, 'raw': a, 'full': b})]
    pass

  dg = RLDExplorer(mis)
  dg.sv.set('vmin', auto_refresh=False)
  dg.sv.set('vmax', auto_refresh=False)
  dg.show()