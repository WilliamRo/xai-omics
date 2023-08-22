from tframe import console



def probe(trainer):
  from tframe.trainers.trainer import Trainer
  from uld.uld_config import ULDConfig
  from uld.uld_set import ULDSet

  # Sanity check
  th = trainer.th
  assert isinstance(trainer, Trainer) and isinstance(th, ULDConfig)

  # Get indices from th
  val_set: ULDSet = trainer.test_set

  # Take snapshot
  val_set.snapshot(trainer.model)

  return 'Snapshot saved to checkpoint folder'
