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


def get_random_window(arr: np.ndarray, window_size=128, slice_size=16,
                      true_rand=False):
  # for Gamma test
  # arr = np.where(arr != 0, 1, arr)
  # s = np.random.randint(arr.shape[1] - slice_size + 1)
  index = np.random.randint(arr.shape[0])

  arr = arr != 0
  arr = arr[index, :, :, :, 0]

  s_arr = np.any(arr, axis=(1, 2))
  h_arr = np.any(arr, axis=(0, 2))
  w_arr = np.any(arr, axis=(0, 1))

  distr_s = normalize(s_arr.ravel())
  distr_w = normalize(h_arr.ravel())
  distr_h = normalize(w_arr.ravel())

  s = windows_choose(distr_s, slice_size)
  h = windows_choose(distr_h, window_size)
  w = windows_choose(distr_w, window_size)

  return index, s, h, w


def get_sample(arr: np.ndarray, index, s, h, w,
               windows_size=128, slice_size=16):

  return arr[index:index+1, s:s+slice_size,
             h:h+windows_size, w:w+windows_size, :]


def gen_windows(arr1: np.ndarray, arr2: np.ndarray, batch_size,
                windows_size=128, slice_size=16, true_rand=False):
  features = []
  targets = []
  for _ in range(batch_size):
    index, s, h, w = get_random_window(arr1, windows_size, slice_size, true_rand)
    features.append(get_sample(arr1, index, s, h, w, windows_size, slice_size))
    targets.append(get_sample(arr2, index, s, h, w, windows_size, slice_size))
  features = np.concatenate(features)
  targets = np.concatenate(targets)

  return features, targets


def nonlocal_mean(image, h=10, patch_size=7, patch_dis=21):
  de_img = restoration.denoise_nl_means(image, h=h,
                                        patch_size=patch_size,
                                        patch_distance=patch_dis)

  return de_img





if __name__ == '__main__':
  from xomics import MedicalImage
  from xomics.objects.general_mi import GeneralMI
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer
  import datetime

  test_gen_win = True

  test = GeneralMI.get_test_sample(r'../../data/02-RLD/rld_data.csv')
  test.process_param['shape'] = [440, 440, 480]
  test.process_param['norm'] = 'PET'

  a = test.images['30G'][0]
  b = test.labels['240G'][0]
  if test_gen_win:
    a = np.expand_dims(a, axis=[-1, 0])
    b = np.expand_dims(b, axis=[-1, 0])

    num = 16
    time1 = datetime.datetime.now()
    img_f, img_t = gen_windows(a, b, num, true_rand=False)
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