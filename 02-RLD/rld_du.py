from rld.rld_agent import RLDAgent, RLDSet
from roma import console


def load_data():
  from rld_core import th

  datasets = RLDAgent.load(th.data_dir, th.val_size, th.test_size)

  # Report data shape
  console.show_info('Data details')
  for ds in datasets:
    assert isinstance(ds, RLDSet)
    if not th.rehearse and ds.name != 'Train-Set' and th.train or ds.name == 'Test-Set':
      ds.fetch_data(ds)
      console.supplement(f'{ds.name}: {ds.features.shape}', level=2)
    else:
      console.supplement(f'{ds.name}: {len(ds)}', level=2)

  return datasets
