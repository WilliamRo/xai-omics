from xomics import MedicalImage
from xomics.data_io.reader.rld_reader import RLDReader
from dev.explorers.rld_explore.rld_explorer import RLDExplorer

data_dir = r'../../../data/02-RLD'
subject = [8, 9]

types = [
  # ['PET', 'WB', '20S', 'STATIC'],
  ['PET', 'WB', '30S', 'GATED'],
  # ['PET', 'WB', '120S', 'STATIC'],
  ['PET', 'WB', '240S', 'GATED'],
  # ['PET', 'WB', '240S', 'STATIC'],
  ['CT', 'WB'],
]

reader = RLDReader(data_dir)

mis = reader.load_data(subject, types, methods='sub', suv=True, raw=True,
                       shape=[256, 440, 440], norm_types=['PET'])

re = RLDExplorer(mis)
re.sv.set('vmin', auto_refresh=False)
re.sv.set('vmax', auto_refresh=False)
re.show()
