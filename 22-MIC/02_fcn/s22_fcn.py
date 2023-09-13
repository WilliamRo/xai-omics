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

s.register('lr', 0.0001, 0.001, 0.01)
s.register('batch_size', 32, 128, 256)
s.register('archi_string', 'fcn-3', 'fcn-4', 'fcn-5')

s.register('random_translation', s.true_and_false)
s.register('random_flip', s.true_and_false)
s.register('random_rotation', s.true_and_false)
s.register('random_noise', s.true_and_false)

# s.configure_engine(times=2)
s.configure_engine(strategy='skopt', criterion='Test Accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)