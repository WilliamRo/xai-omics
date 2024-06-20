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
# dose = s.common_parameters['dose']

gpu_id = 0
summ_name = s.default_summ_name # + '_' + dose

s.register('gpu_id', gpu_id)
s.register('gather_summ_name', summ_name + '.sum')
s.register('allow_growth', False)
s.register('probe_cycle', 0)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 10)

s.register('dose', '1-50')

# s.register('data_shape', [1, 608, 440, 440, 1])

s.register('beta', 0, 0.02, hp_type=tuple)
s.register('archi_string', '3-3-3', '3-3-3-3')
s.register('sigmas', '1-2-4')
s.register('use_bias', s.true_and_false)
s.register('thickness', 20)



s.configure_engine(criterion='Best NRMSE', greater_is_better=False)
s.configure_engine(add_script_suffix=True)
# s.configure_engine(strategy='skopt')
s.configure_engine(times=5)
s.run(rehearsal=0)
