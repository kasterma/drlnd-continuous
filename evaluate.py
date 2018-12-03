import numpy as np
import matplotlib.pyplot as plt

dat = np.load("scores-0.npy")
plt.plot(dat)
plt.show()
