import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from mic_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure datas set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 2000)
s.register('patience', 10)

s.register('lr', 0.0005625248714883541)
s.register('batch_size', 32)
s.register('archi_string', 'fcn-4')

s.register('random_translation', False)
s.register('random_flip', False)
s.register('random_rotation', False)
s.register('random_noise', False)

s.configure_engine(times=10, add_script_suffix=True)
# s.configure_engine(strategy='skopt', criterion='Test Accuracy',
#                    greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)