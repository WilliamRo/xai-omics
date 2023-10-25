from xomics.data_io.rld_reader import RLDReader
from dev.explorers.rld_explore.rld_explorer import RLDExplorer

data_dir = r'../../../data/02-RLD'
subject = [0, 12]

types = [
  ['CT', 'WB'],
  ['PET', 'WB', '240S', 'GATED'],
  ['PET', 'WB', '240S', 'STATIC'],
]

reader = RLDReader(data_dir)

mis = reader.load_data(subject, types, methods='mi', raw=True, suv=True)

re = RLDExplorer(mis)
re.sv.set('vmin', auto_refresh=False)
re.sv.set('vmax', auto_refresh=False)
re.show()
