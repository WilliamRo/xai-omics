
import rld_core as core
import rld_mu as m

from tframe import console, SaveMode
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gan'
id = 7
def model():
  th = core.th
  gan = m.get_gan_container()

  m.mu.UNet(3, arc_string=th.archi_string).add_to(gan.G)

  # H, N = 8, 3
  # for _ in range(N):
  #   gan.D.add(m.mu.HyperConv3D(H, 3, use_batchnorm=True, activation='lrelu'))
  #   H *= 2
  # gan.D.add(m.mu.Dense(H, activation='lrelu'))
  m.mu.UNet(3, arc_string=th.archi_string, use_batchnorm=True).add_to(gan.D)


  return m.gan_finalize(gan)


def main(_):
  console.start('{} on PET/CT reconstruct task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.visible_gpu_id = 0
  th.data_config = fr'alpha dataset=02-RLD'

  th.val_size = 5
  th.test_size = 10

  th.windows_size = [16, 128, 128]
  # th.eval_windows_size = [1, 128, 128]

  th.data_shape = [560, 440, 440]
  # th.input_shape = th.input_shape[1:]

  th.gan = True

  th.data_set = ['30G', '240G']
  th.process_param = {
    'ct_window': None,
    'norm': 'PET',  # only min-max,
    'shape': th.data_shape[::-1],  # [320, 320, 240]
    'crop': [10, 0, 0][::-1],  # [30, 30, 10]
    'clip': None,  # [1, None]
  }

  th.noCT = True
  if th.noCT:
    th.input_shape[-1] = 1
  # th.use_suv = False

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.suffix += f'_win{tuple(th.windows_size)}'


  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '4-3-3-2-lrelu'

  th.use_sigmoid = False
  th.clip_off = True
  th.output_conv = False
  # th.use_res = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = 0
  # th.validation_per_round = 50

  th.batch_size = 4
  th.batchlet_size = 2
  th.val_batch_size = 2

  th.buffer_size = 6

  th.loss_string = 'nrmse'
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.suffix += '_noCT' if th.noCT else ''
  th.suffix += f'_{th.data_set[0]}to{th.data_set[1]}'
  th.suffix += f'_lr{th.learning_rate}_bs{th.batch_size}_{th.opt_str}'
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
