from xomics.data_io.uld_reader import UldReader
from xomics.data_io.utils.uld_raw_rd import rd_uld_test
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage

data_dir = r'../../../data/01-ULD/'
subjects = [1, 12]
mode = "uld-train"


keys = ['Full',
        # '1-2',
        '1-4',
        # '1-10',
        # '1-20',
        # '1-50',
        # '1-100',
        ]
mis = []
reader = UldReader(data_dir)

if mode == 'uld-train':
  mis = reader.load_mi_data(subjects, keys)

if mode == 'uld-pair':
  doses = {
    "feature": keys[1],
    "target": keys[0]
  }
  img_f, img_l = reader.load_data(subjects, doses)
  for i in range(img_f.shape[0]):
    mi = MedicalImage(f'Subject-i', {'Full': img_f[i], 'Low': img_l[i]})
    mis.append(mi)

if mode == 'uld-test':
  imgs = rd_uld_test(data_dir + "testset", 3)
  for i in range(len(imgs)):
    mi = MedicalImage(f"test-{i}", {'low-test': imgs[i]})
    mis.append(mi)

# Visualization
dg = DrGordon(mis)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
