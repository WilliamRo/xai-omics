from mi.mi_agent import MIAgent
from mi.mi_set import MISet


load_data = MIAgent.load

if __name__ == '__main__':
  from mi_core import th

  ds: MISet = load_data()