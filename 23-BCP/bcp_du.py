from bcp.bcp_agent import BCPAgent
from bcp.bcp_set import BCPSet


load_data = BCPAgent.load


if __name__ == '__main__':
  from bcp_core import th

  ds: BCPSet = load_data()
  ds[0].visualize_self(20)