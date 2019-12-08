import numpy as np

functions = {'sin': lambda deg, x: ((np.sin(x) + np.pi) / (2 * np.pi)) ^ deg,
             'cos': lambda deg, x: (np.cos(x) + np.pi) / (2 * np.pi),
             'arctg': lambda deg, x: (np.arctan(x) + np.pi / 2) / np.pi}
