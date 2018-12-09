from drlnd_continuous.noise import OUNoise
import numpy as np
import random
import copy


def test_run_noise():
    np.random.seed(42)
    noise = OUNoise(2)
    assert noise.sample().shape == (2,)

    noise = OUNoise(6)
    assert noise.sample().shape == (6,)

    # the next bit of the test was used to change the setup in the noise to use np.random; all we needed (we think) was
    # to make sure the mean and stdev didn't change.  Though with no real understanding of the process or why we use
    # it, something important could have been missed.
    dat = np.array([noise.sample() for _ in range(10_000)])
    print("Mean is {}".format(np.mean(dat, axis=0)))
    print("Stdev is {}".format(np.std(dat, axis=0)))

class OUNoise2:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() - 0.5 for i in range(len(x))])
        self.state = x + dx
        return self.state


def plot_noise():
    """Draw some of the noise to see the development"""
    import matplotlib.pyplot as plt   # put here b/c matplotlib needs framework python

    np.random.seed(41)
    noise = OUNoise(2)
    dat = np.array([noise.sample() for _ in range(1_000)])
    plt.plot(dat)
    plt.show()
    plt.plot(dat[:,0], dat[:,1])
    plt.show()
