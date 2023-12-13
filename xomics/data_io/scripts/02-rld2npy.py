from xomics.data_io.utils.raw_rw import *

import os


sub_ids = ['10217', '10163', '10146', '10103', '10104']
del_ids = ['10665', '10488', '10632', '10717']


def rld2npy(datadir, new_dir):
  l1_dirs = sorted(os.listdir(datadir))
  num = len(l1_dirs)
  ids = 0
  id_180 = 0
  id_del = 0
  for sub, l1 in enumerate(l1_dirs):
    if sub != 0:
      return
    l1_path = os.path.join(datadir, l1)
    l2_dir = os.listdir(l1_path)[0]
    l2_path = os.path.join(l1_path, l2_dir)
    image_dirs = sorted(os.listdir(l2_path))
    print(f'Reading data from {l1} ({sub}/{num}):')
    ctwb = None
    ctbh = None
    flag = False
    for file in image_dirs:
      print(f'...Reading images from {file}')
      image_path = os.path.join(l2_path, file)

      tag = get_tags(image_path, suv=False, isSeries=True)

      name_list = file.split('_')
      func3 = lambda x, y: x if x in name_list else y

      img_type = func3('CT', 'PET')
      pos = func3('WB', 'BH')

      if img_type == 'CT':
        if pos == 'WB':
          ctwb = rd_series_itk(image_path)
        elif pos == 'BH':
          ctbh = rd_series_itk(image_path)
        image = rd_series_itk(image_path)
      elif img_type == 'PET':
        if pos == 'WB':
          image = rd_series_itk(image_path) # , resample=True, refimage=ctwb)
        elif pos == 'BH':
          image = rd_series_itk(image_path) # , resample=True, refimage=ctbh)

      # image = image.reshape((1,) + image.shape + (1,))

      stime = ''
      times = [15, 20, 30, 40, 60, 120, 180, 240]
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

      index = f'sub{ids}'
      for i in sub_ids:
        if i in l1:
          index = f'180S_sub{id_180}'
          flag = True
          break

      for i in del_ids:
        if i in l1:
          index = f'del_sub{id_del}'
          flag = True
          break


      name = f'{index}_{img_type}_{pos}{stime}{control}{param}'
      # print(f'<{name}>')
      sub_path = os.path.join(new_dir, f'{index}')
      img_path = os.path.join(sub_path, f'{name}.nii.gz')
      tag_path = os.path.join(sub_path, f'{name}.pkl')
      if not os.path.exists(sub_path):os.mkdir(sub_path)
      # return
      sitk.WriteImage(image, img_path)
      wr_tags(tag, tag_path)
      with open(os.path.join(sub_path, f'{l1}.NAME'),'w') as f:
        pass
      print(f'...Saved image and tag to {sub_path}/{name}')
    if flag:
      if '180S' in index:
        id_180 += 1
      elif 'del' in index:
        id_del += 1
    else:
      ids += 1


if __name__ == '__main__':
  datadir = 'F:\\xai-omics-data\\02-RLD-RAW\\'
  # datadir = 'D:\\projects\\xai-omics\\data\\02-RLD-RAW\\'
  new_dir = 'D:\\projects\\xai-omics\\data\\02-RLD\\'
  # rd_series('D:\\projects\\xai-omics\\data\\02-RLD-RAW\\CHEN_HE_PING_YHP00011233\\PET_13_DL_WB_GATED_(ADULT)_20230906_111759_623000\\CT_BH_15S_3_0_B30F_0013')
  rld2npy(datadir, new_dir)


