import numpy as np



# self.features/targets.shape = [N, S, H, W, 2]
def get_center(arr: np.ndarray, size):
  start_x = (arr.shape[-2] - size) // 2
  start_y = (arr.shape[-3] - size) // 2

  return arr[:, :, start_x:start_x + size, start_y:start_y + size]


def normalize(arr: np.array):
  norm = np.linalg.norm(arr, ord=1)
  return arr / norm


def windows_choose(distr: np.ndarray, windows_size):
  x = np.linspace(0, distr.shape[0] - 1, distr.shape[0])
  result = np.random.choice(x, p=distr)
  result = result - windows_size / 2

  if result < 0: result = 0
  if result > distr.shape[0] - windows_size:
    result = distr.shape[0] - windows_size

  return int(result)


def get_random_window(arr: np.ndarray, window_size=128, slice_size=16,
                      true_rand=True):
  # for Gamma test
  # arr = np.where(arr != 0, 1, arr)
  s = np.random.randint(arr.shape[1] - slice_size + 1)
  index = np.random.randint(arr.shape[0])

  if true_rand:
    h = np.random.randint(arr.shape[2] - window_size + 1)
    w = np.random.randint(arr.shape[3] - window_size + 1)
  else:
    arr = arr[index:index+1]
    arr = np.where(arr != 0, 1, arr)
  # arr_pro = np.add.reduce(arr, axis=2)
  # distr_s = normalize(np.add.reduce(arr_pro, axis=2).reshape((-1)))
    arr = arr[:, s:s+slice_size]
    arr_pro = np.add.reduce(arr, axis=1)
    arr_pro = np.add.reduce(arr_pro, axis=3)
    distr_w = normalize(np.add.reduce(arr_pro, axis=1).reshape((-1)))
    distr_h = normalize(np.add.reduce(arr_pro, axis=2).reshape((-1)))
  # print(h,w)
  # s = windows_choose(distr_s, slice_size)
    h = windows_choose(distr_h, window_size)
    w = windows_choose(distr_w, window_size)

  return index, s, h, w


def get_sample(arr: np.ndarray, index, s, h, w,
               windows_size=128, slice_size=16):

  return arr[index:index+1, s:s+slice_size,
             h:h+windows_size, w:w+windows_size, :]


def gen_windows(arr1: np.ndarray, arr2: np.ndarray, batch_size,
                windows_size=128, slice_size=16, true_rand=True):
  features = []
  targets = []
  for _ in range(batch_size):
    index, s, h, w = get_random_window(arr1, windows_size, slice_size, true_rand)
    features.append(get_sample(arr1, index, s, h, w, windows_size, slice_size))
    targets.append(get_sample(arr2, index, s, h, w, windows_size, slice_size))
  features = np.concatenate(features)
  targets = np.concatenate(targets)

  return features, targets




if __name__ == '__main__':
  from xomics import MedicalImage
  from xomics.data_io.uld_reader import UldReader
  from xomics.gui.dr_gordon import DrGordon

  path = '../'
  reader = UldReader(path)
  a = reader.load_data([1, 5, 12], "1-4")
  b = reader.load_data([1, 5, 12], "Full")
  num = 25
  img_f, img_t = gen_windows(a, b, num, true_rand=False)
  # print(len(img))
  # print(a.shape, img[0].shape)
  di_f = {}
  di_t = {}
  for i in range(num):
    di_f[f'feature-{i}'] = img_f[i]
  for i in range(num):
    di_t[f'target-{i}'] = img_t[i]
  mi_f = MedicalImage('feature', di_f)
  mi_t = MedicalImage('target', di_t)
  dg = DrGordon([mi_f, mi_t])
  dg.slice_view.set('vmin', auto_refresh=False)
  dg.slice_view.set('vmax', auto_refresh=False)
  dg.show()
  # print(get_center(arr, 128).shape)
  # prob = calc_prob(arr, 128)
  # print(prob.shape)
