from roma import finder
from xomics.data_io.utils.raw_rw import npy_save, wr_tags
from xomics.data_io.utils.uld_raw_rd import rd_uld_train

import os



def gen_uld_npy(path, n_path):
  subjects = finder.walk(path, type_filter='folder', pattern='Subject_*',
                         return_basename=True)

  for count, subject in enumerate(subjects):
    print(f"({count + 1}/{len(subjects)}) Reading {subject}...")

    c1 = 0
    for dose in doses:
      c1 += 1
      data, tags = rd_uld_train(path, subject, dose=dose)
      num = int(subject[8:].split('-')[0])
      print(f"..({c1}/{len(doses)}) Reading {dose}")
      dose = dose[:-5]
      c2 = 0
      for arr, tag in zip(data, tags):
        c2 += 1
        # print(arr.shape, tag)
        npypath = os.path.join(n_path,
                               f'subject{num}', f'subject{num}_{dose}.npy')
        tagpath = os.path.join(n_path,
                               f'subject{num}', f'tags_subject{num}_{dose}.txt')
        npy_save(arr, npypath)
        print(f"....({c2}/{len(data) * 2}) Save numpy data subject{num} {dose}")
        wr_tags(tag, tagpath, suv=True)
        c2 += 1
        print(f"....({c2}/{len(data) * 2}) Save tags data subject{num} {dose}")
        num += 1


def test_uld2npy():
  pass




if __name__ == '__main__':
  doses = [
    'Full_dose',
    '1-2 dose',
    '1-4 dose',
    '1-10 dose',
    '1-20 dose',
    '1-50 dose',
    '1-100 dose',
  ]
  path = "../../../data/01-ULD-RAW/"
  n_path = "../../../data/01-ULD/"
  gen_uld_npy(path, n_path)
