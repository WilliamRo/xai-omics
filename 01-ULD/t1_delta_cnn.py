import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.layers.merge import Merge
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'delta'
id = 3

class WeightedSum(Merge):
  full_name = 'weighted_sum'
  abbreviation = 'ws'

  def __init__(self, w):
    self.w = w
    self.abbreviation += f'{w}'
    self.full_name += f'{w}'

  def _link(self, x_list, **kwargs):
    assert len(x_list) == 2
    return x_list[0] + x_list[1] * self.w


def model():
  mu = m.mu
  th = core.th

  model = m.get_initial_model()

  conv = lambda n: mu.HyperConv3D(
    filters=n, kernel_size=th.kernel_size, activation=th.activation)

  vertices = [
    [conv(8)],
    [conv(1)],
    [WeightedSum(1.0)],
    # [mu.Activation('sigmoid')]
  ]
  edges = '1;01;101'
  fm = m.mu.ForkMergeDAG(vertices, edges, name='Delta')
  model.add(fm)

  model.build(loss=m.get_pw_rmse(), metric=[
    m.get_ssim_3D(), m.get_nrmse(), m.get_psnr(), 'loss'])

  return model


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = r'delta dataset=01-ULD dose=1-4'

  th.val_size = 30
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128

  th.slice_num = 608
  th.use_tanh = 0
  th.use_color = False
  th.use_suv = False
  th.norm_by_feature = True
  th.train_self = not th.norm_by_feature

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_self'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '4-3-3-2-lrelu'
  th.kernel_size = 5
  th.activation = 'lrelu'

  th.learn_delta = 1
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = th.updates_per_round

  th.batch_size = 1
  th.val_batch_size = 1

  th.buffer_size = 18

  th.loss_string = 'rmse'
  th.optimizer = 'adam'
  # th.optimizer = 'sgd'
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}'.format(model_name)
  if th.use_tanh != 0: th.mark += f'-tanh({th.use_tanh})'

  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
