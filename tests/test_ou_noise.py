from drlnd_continuous.noise import OUNoise
import numpy as np


def test_run_noise():
    noise = OUNoise(2, seed=42)
    assert noise.sample().shape == (2,)

    noise = OUNoise(6, seed=42)
    assert noise.sample().shape == (6,)

    # the next bit of the test was used to change the setup in the noise to use np.random; all we needed (we think) was
    # to make sure the mean and stdev didn't change.  Though with no real understanding of the process or why we use
    # it, something important could have been missed.
    dat = np.array([noise.sample() for _ in range(10_000)])
    print("Mean is {}".format(np.mean(dat, axis=0)))
    print("Stdev is {}".format(np.std(dat, axis=0)))
