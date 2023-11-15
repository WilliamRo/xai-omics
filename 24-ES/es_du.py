from es.es_agent import ESAgent
from es.es_set import ESSet


load_data = ESAgent.load


if __name__ == '__main__':

  train_set, val_set, test_set = load_data()
  train_set.visualize_self(100)


