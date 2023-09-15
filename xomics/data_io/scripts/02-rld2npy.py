from xomics.data_io.utils.raw_rw import *

import os


sub_ids = [18, 23, 33, 43, 45]

def rld2npy(datadir, new_dir):
  l1_dirs = os.listdir(datadir)
  num = len(l1_dirs)
  ids = 0
  id_180 = 0
  for sub, l1 in enumerate(l1_dirs):
    l1_path = os.path.join(datadir, l1)
    l2_dir = os.listdir(l1_path)[0]
    l2_path = os.path.join(l1_path, l2_dir)
    image_dirs = os.listdir(l2_path)
    print(f'Reading data from {l1} ({sub}/{num}):')
    flag = False
    for file in image_dirs:
      print(f'...Reading images from {file}')
      image_path = os.path.join(l2_path, file)
      image = rd_series(image_path)
      image = image.reshape((1,) + image.shape + (1,))
      tag = get_tags(image_path, suv=False, isSeries=True)

      name_list = file.split('_')
      func3 = lambda x, y: x if x in name_list else y

      img_type = func3('CT', 'PET')
      pos = func3('WB', 'BH')

      stime = ''
      times = [15, 20, 30, 120, 180, 240]
      for sec in times:
        if f'{sec}S' in name_list:
          stime = f'_{sec}S'
          break
      control = ''
      ctrl_type = ['GATED', 'STATIC']
      for ctrl in ctrl_type:
        if ctrl in name_list:
          control = '_' + ctrl
          break

      param = ''
      param_type = ['2I5S', '4I5S']
      for par in param_type:
        if par in name_list:
          param = '_' + par
          break

      if sub in sub_ids:
        index = f'180S_sub{id_180}'
        flag = True
      else:
        index = f'sub{ids}'

      name = f'{index}_{img_type}_{pos}{stime}{control}{param}'
      # print(f'<{name}>')
      sub_path = os.path.join(new_dir, f'{index}')
      img_path = os.path.join(sub_path, f'{name}.npy')
      tag_path = os.path.join(sub_path, f'{name}.pkl')

      # return
      npy_save(image, img_path)
      wr_tags(tag, tag_path)
      print(f'...Saved image and tag to {sub_path}/{name}')
    if flag:
      id_180 += 1
    else:
      ids += 1


if __name__ == '__main__':
  datadir = 'D:\\projects\\xai-omics\\data\\02-RLD-RAW\\'
  new_dir = '/data/02-RLD\\'
  # rd_series('D:\\projects\\xai-omics\\data\\02-RLD-RAW\\CHEN_HE_PING_YHP00011233\\PET_13_DL_WB_GATED_(ADULT)_20230906_111759_623000\\CT_BH_15S_3_0_B30F_0013')
  rld2npy(datadir, new_dir)


