from rld.rld_agent import RLDAgent, RLDSet
from roma import console
from tframe import pedia


def load_data(model):
  from rld_core import th

  datasets: list[RLDSet] = RLDAgent.load(th.data_dir, th.val_size, th.test_size)

  # Report data shape
  console.show_info('Data details')
  for ds in datasets:
    if hasattr(model, 'batch_preprocessor'):
      ds.batch_preprocessor = model.batch_preprocessor
    assert isinstance(ds, RLDSet)

    if not th.rehearse and (ds.name != 'Train-Set' and th.train or ds.name == 'Test-Set'):
      ds.mi_data.pre_load(16)
      ds.fetch_data(ds)
      console.supplement(f'{ds.name}: {ds.features.shape}', level=2)
    else:
      # if th.train and not th.rehearse:
      #   ds.mi_data.pre_load(48)
      console.supplement(f'{ds.name}: {len(ds)}', level=2)

  if th.gan:
    datasets[0].data_dict[pedia.D_input] = datasets[0].targets
    for i in range(len(datasets)):
      datasets[i].data_dict[pedia.G_input] = datasets[i].targets
  return datasets
