# Training driver for the DRLND continuous project
#
# This code is the primary interface to start and evaluate the training for the continous project.

from drlnd_continuous import *
import numpy as np


def random_test_run():
    """Take 100 randomly generated steps in the environment"""
    env = Reacher()
    env.reset(train_mode=False)
    for idx in range(100):
        # noinspection PyUnresolvedReferences
        act_random = np.clip(np.random.randn(20, 4), -1, 1)
        step_result = env.step(act_random)
        if step_result.done:
            break


random_test_run()
