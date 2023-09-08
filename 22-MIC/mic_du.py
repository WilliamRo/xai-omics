from mic.mic_agent import MICAgent
from mic.mic_set import MICSet


load_data = MICAgent.load


if __name__ == '__main__':
  from mic_core import th

  ds: MICSet = load_data()
  ds[0].visualize_self(20)