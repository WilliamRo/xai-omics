import sys
sys.path.append('../')
sys.path.append('../../')


from rld.rld_config import RLDConfig
from tframe.utils.script_helper import Helper


Helper.register_flags(RLDConfig)
s = Helper()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass
s.register('rehearse', 1)
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
gpu_id = 1
summ_name = s.default_summ_name

s.register('gpu_id', gpu_id)
s.register('gather_summ_name', summ_name + '.sum')
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 15)

# s.register('window_size', , 128)
# s.register('slice_size', 32, 128)

s.register('lr', 0.0001, 0.1)

# s.register('batch_size', 1, 8)


s.configure_engine(strategy='skopt', criterion='Best PSNR',
                   greater_is_better=True)
s.run(rehearsal=0)
