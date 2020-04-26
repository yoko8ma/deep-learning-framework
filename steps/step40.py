import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)