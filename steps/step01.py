class Varable:
    def __init__(self, data):
        self.data = data

import numpy as np

data = np.array(1.0)
x = Varable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)