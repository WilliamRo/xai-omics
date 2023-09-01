import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from uld_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.9)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 20)

s.register('window_size', 64, 128)
s.register('slice_size', 64, 128)
s.register('data_config', r'epsilon dataset=01-ULD dose=1-4')
# s.register('data_shape', [1, 608, 440, 440, 1])
s.register('norm_by_feature', True)

s.register('archi_string', '4-3-3-2-lrelu')
# s.register('rand_batch', True, False)
# s.register('learn_delta', False, True)

s.register('developer_code', 'adam', 'sgd')
s.register('lr', 0.0005, 0.003)

s.register('batch_size', 1, 4)
s.register('val_batch_size', 1)
s.register('buffer_size', 18)
s.register('val_decimals', 7)
s.register('loss_string', 'rmse', 'pw_rmse')


s.configure_engine(strategy='skopt', criterion='Best F1')
s.configure_engine(times=2)
s.run(rehearsal=1)
