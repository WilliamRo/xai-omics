import sys
sys.path.append('../')
sys.path.append('../../')


from uld.uld_config import ULDConfig
from tframe.utils.script_helper import Helper


Helper.register_flags(ULDConfig)
s = Helper()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
dose = s.common_parameters['dose']

gpu_id = 0
summ_name = s.default_summ_name + '_' + dose

s.register('gpu_id', gpu_id)
s.register('gather_summ_name', summ_name + '.sum')
s.register('allow_growth', True)
s.register('probe_cycle', 0)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 15)

# s.register('dose', '1-4', '1-20', '1-50', '1-100')
# s.register('window_size', 64, 128)
# s.register('slice_size', 64, 128)
# s.register('dose', dose)

# s.register('data_shape', [1, 608, 440, 440, 1])
s.register('norm_by_feature', True)

s.register('archi_string', '4-3-3-2-lrelu')
s.register('rand_batch', True)
# s.register('learn_delta', False, True)

s.register('opt_str', 'adam', 'sgd')
s.register('lr', 0.0005, 0.003)

s.register('batch_size', 4, 8)
s.register('val_batch_size', 3)
s.register('buffer_size', 18)
s.register('loss_string', 'rmse')


s.configure_engine(strategy='skopt', criterion='Best SSIM',
                   greater_is_better=True)
s.run(rehearsal=0)
