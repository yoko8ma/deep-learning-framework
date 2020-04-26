import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functons as F
from dezero.utils import plot_dot_graph


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)