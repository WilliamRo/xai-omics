from tframe import console, Predictor
from tframe.trainers.trainer import Trainer


def probe(trainer: Trainer):
  from tframe import context
  from tframe.core import TensorSlot
  from rld_core import th

  set_region_alpha = context.depot['set_internal_alpha']
  loss_lesion_slot = [
    slot for slot in list(trainer.batch_loss_stats.keys()) if 'internal' in slot.name]
  assert len(loss_lesion_slot) == 1
  loss_lesion_slot = loss_lesion_slot[0]
  assert isinstance(loss_lesion_slot, TensorSlot)

  alpha = 0.9
  if trainer.batch_loss_stats[loss_lesion_slot].running_average/alpha < 0.5:
    alpha = 0.3
  set_region_alpha(alpha)

  return f'Current alpha: {alpha}'
