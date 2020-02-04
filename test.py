import numpy as np
import matplotlib.pyplot as plt

x = np.asarray([1, 2, 2, 4]).reshape(2, 2)
y = np.linalg.inv(x)

print(y)

np.linalg.cond()