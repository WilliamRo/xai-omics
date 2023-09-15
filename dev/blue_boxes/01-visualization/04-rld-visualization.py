from xomics.data_io.rld_reader import RLDReader
from xomics.gui.dr_gordon import DrGordon

data_dir = r'../../../data/02-RLD'
subject = [0, 12]

types = [
  ['CT', 'WB'],
  ['PET', 'WB', '240S', 'GATED'],
  # ['PET', 'WB', '30S', 'GATED'],
]

reader = RLDReader(data_dir)

mis = reader.load_data(subject, types, methods='mi', raw=True)

dg = DrGordon(mis)
dg.slice_view.set('vmin', auto_refresh=False)
dg.slice_view.set('vmax', auto_refresh=False)
dg.show()
