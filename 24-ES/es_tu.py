import random
import numpy as np

from tframe import console, Predictor
from tframe.trainers.trainer import Trainer
from es.es_config import ESConfig
from es.es_set import ESSet
from collections import OrderedDict



def probe(trainer: Trainer):
  from tframe import context
  from tframe.core import TensorSlot
  from es_core import th

  set_region_alpha = context.depot['set_region_alpha']
  loss_lesion_slot = [
    slot for slot in list(trainer.batch_loss_stats.keys()) if 'lesion' in slot.name]
  assert len(loss_lesion_slot) == 1
  loss_lesion_slot = loss_lesion_slot[0]
  assert isinstance(loss_lesion_slot, TensorSlot)

  if (trainer.batch_loss_stats[loss_lesion_slot].running_average < 0.3
      and th.alpha_var == 0.0):
    set_region_alpha(th.alpha_region)
    th.alpha_var = th.alpha_region

  return f'Current alpha_region: {th.alpha_var}'


def evaluate(trainer: Trainer):
  from es_core import th

  model: Predictor = trainer.model
  agent = model.agent

  evaluate_pro = lambda ds: model.validate_model(
    ds, batch_size=th.eval_batch_size, allow_sum=False,
    verbose=th.val_progress_bar)

  test_set = trainer.test_set

  scalar_dict_test = evaluate_pro(test_set)

  for slot, value in scalar_dict_test.items():
    agent.put_down_criterion(test_set.name + ' ' + slot.symbol, value)