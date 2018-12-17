# code to draw the episode score graphs for in the report

import numpy as np
import matplotlib.pyplot as plt

dat = np.load("scores-15.npy")
plt.plot(dat, label="train")
dat = np.load("evaluate-scores-15.npy")
plt.plot(dat, label="eval")
plt.legend()
#plt.show()    # use this during development to show (not save) the graph
plt.savefig("both-scores.png")
