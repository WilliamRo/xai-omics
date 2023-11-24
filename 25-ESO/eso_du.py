from eso.eso_agent import ESOAgent


load_data = ESOAgent.load


if __name__ == '__main__':

  train_set, val_set, test_set = load_data()
  train_set.visualize_self(100)


