import os

from tframe import console, Predictor
from tframe.trainers.trainer import Trainer
from tframe.utils import imtool
from collections import OrderedDict

from xomics.data_io.utils.metrics_calc import get_metrics


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
  if trainer.batch_loss_stats[loss_lesion_slot].running_average/alpha < 0.4:
    alpha = 0.6
  set_region_alpha(alpha)

  return f'Current alpha: {alpha}'


def save_img(trainer: Trainer):
  import matplotlib.pyplot as plt
  SAMPLE_NUM = 4

  data = trainer.validation_set.features[:1]
  targets = trainer.validation_set.targets[:1]

  model = trainer.model
  samples = model.generate(data, sample_num=SAMPLE_NUM)

  csv_path = os.path.join(
    model.agent.ckpt_dir, f'metrics.csv')

  target_path = os.path.join(
    model.agent.ckpt_dir, f'z-targets.png')
  if not os.path.exists(target_path):
    fig = imtool.gan_grid_plot(targets[0, 145:145+SAMPLE_NUM, ..., 0])
    plt.savefig(target_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)

  feature_path = os.path.join(
    model.agent.ckpt_dir, f'z-features.png')
  if not os.path.exists(feature_path):
    metrics = get_metrics(targets[0, ..., 0], data[0, ..., 0], ['nrmse', 'ssim', 'psnr'])
    metrics = ["{:.4f}".format(num) for num in metrics.values()]

    with open(csv_path, 'w') as f:
      f.write(','.join(metrics)+'\n')
    fig = imtool.gan_grid_plot(data[0, 145:145+SAMPLE_NUM, ..., 0])
    plt.savefig(feature_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)

  metrics = get_metrics(targets[0, ..., 0], samples[0, ..., 0], ['nrmse', 'ssim', 'psnr'])
  metrics = ["{:.4f}".format(num) for num in metrics.values()]

  with open(csv_path, 'a') as f:
    f.write(','.join(metrics)+'\n')

  file_path = os.path.join(
    model.agent.ckpt_dir, f'round-{trainer.total_rounds:.1f}.png')
  fig = imtool.gan_grid_plot(samples[0, 145:145+SAMPLE_NUM, ..., 0])
  plt.savefig(file_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
  plt.close(fig)

  # Take notes for export
  scalars = OrderedDict(
    {k.name: v.running_average for k, v in trainer.batch_loss_stats.items()})
  trainer.model.agent.take_down_scalars_and_tensors(scalars, OrderedDict())
  trainer._inter_cut('Notes taken down.', prompt='[Export]')

  return f'Image saved to `{file_path}`'