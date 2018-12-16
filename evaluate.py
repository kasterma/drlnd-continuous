import numpy as np
import matplotlib.pyplot as plt

dat = np.load("scores-2.npy")
plt.plot(dat)
plt.show()
