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

gpu_id = 1
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

s.register('beta', 0.01, 0.03)




s.configure_engine(strategy='skopt', criterion='Best SSIM',
                   greater_is_better=True)
s.run(rehearsal=1)
