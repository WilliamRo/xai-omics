from uld.uld_agent import ULDAgent, ULDSet
from roma import console
from xomics.data_io.uld_reader import UldReader


def load_data():
  from uld_core import th

  datasets = ULDAgent.load(th.data_dir, th.val_size, th.test_size)

  console.show_info('Data details')

  for ds in datasets:
    assert isinstance(ds, ULDSet)
    ds.fetch_data(ds)
    ds.data_fetcher = None

  return datasets
