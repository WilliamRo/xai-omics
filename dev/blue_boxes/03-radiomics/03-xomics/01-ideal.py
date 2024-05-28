from pictor.xomics.omix import Omix

import numpy as np



x1 = np.random.randint(0, 40, 100)
x2 = np.random.randint(60, 100, 100)
x = np.concatenate([x1, x2]) / 100
x = np.reshape(x, (-1, 1))
y = [0] * 100 + [1] * 100

omix = Omix(x, y, ['Awesome Signature'], ['Low Grade', 'High Grade'])
omix.show_in_explorer(fig_size=(4, 4))
