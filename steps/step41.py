import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)

