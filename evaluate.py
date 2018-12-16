import numpy as np
import matplotlib.pyplot as plt

# dat = np.load("scores-11.npy")
# plt.plot(dat)
# plt.show()
#
# plt.savefig("scores.png")
#
# dat = np.load("evaluate-scores-11.npy")
# plt.plot(dat)
# plt.show()
#
# plt.savefig("eval-scores.png")

dat = np.load("scores-11.npy")
plt.plot(dat, label="train")
dat = np.load("evaluate-scores-11.npy")
plt.plot(dat, label="eval")
plt.legend()
#plt.show()
plt.savefig("both-scores.png")
