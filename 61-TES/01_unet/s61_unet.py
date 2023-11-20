import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from tes_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure datas set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 1

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
s.register('gpu_memory_fraction', 0.8)
s.register('rehearse', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 5000)
s.register('patience', 15)

s.register('random_noise', True)

s.register('lr', 0.01, 0.001, 0.0001, 0.00001)
s.register('batch_size', 8, 16)
s.register('archi_string',
           '8-9-3-3-relu-mp', '8-7-3-3-relu-mp')

# s.configure_engine(times=5)
s.configure_engine(strategy='skopt', criterion='Best dice_accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)

