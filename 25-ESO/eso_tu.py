import random
import numpy as np

from tframe import console, Predictor
from tframe.trainers.trainer import Trainer



def probe(trainer: Trainer):
  pass


def evaluate(trainer: Trainer):
  from eso_core import th

  model: Predictor = trainer.model
  agent = model.agent

  evaluate_pro = lambda ds: model.validate_model(
    ds, batch_size=th.eval_batch_size, allow_sum=False,
    verbose=th.val_progress_bar)

  test_set = trainer.test_set

  scalar_dict_test = evaluate_pro(test_set)

  for slot, value in scalar_dict_test.items():
    agent.put_down_criterion(test_set.name + ' ' + slot.symbol, value)