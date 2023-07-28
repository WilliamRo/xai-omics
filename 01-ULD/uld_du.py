from uld.uld_agent import ULDAgent, ULDSet
from roma import console



def load_data():
  from uld_core import th

  datasets = ULDAgent.load(th.data_dir, th.val_size, th.test_size)
  # datasets = list(datasets)
  #
  # for i in range(1, 3):
  #   datasets[i] = datasets[i].eval_set
  # datasets[1].name = 'Val-Set'
  # datasets[2].name = 'Test-Set'
  #
  # # Report data shape
  # console.show_info('Data details')
  # for ds in datasets:
  #   assert isinstance(ds, ULDSet)
  #   console.supplement(f'{ds.name}: {ds.features.shape})', level=2)

  return datasets
