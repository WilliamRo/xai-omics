from bcp.bcp_agent import BCPAgent
from bcp.bcp_set import BCPSet


load_data = BCPAgent.load


if __name__ == '__main__':
  from bcp_core import th
  import numpy as np

  train_set, valid_set, test_set = load_data()
  train_list = train_set.data_dict['mi_list'].tolist()

  from xomics.gui.dr_gordon import DrGordon
  dg = DrGordon(train_list)
  dg.slice_view.set('vmax', auto_refresh=False)
  dg.slice_view.set('vmin', auto_refresh=False)
  dg.show()

