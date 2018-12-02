# Training driver for the DRLND continuous project
#
# This code is the primary interface to start and evaluate the training for the continous project.

from drlnd_continuous import *
import numpy as np


def random_test_run():
    """Take 100 randomly generated steps in the environment

    In this interaction also interact with the Agent (to test this interaction while developing it)
    """
    env = Reacher()
    agent = Agent(10000)
    state = env.reset(train_mode=False)
    for step_idx in range(100):
        # noinspection PyUnresolvedReferences
        act_random = np.clip(np.random.randn(20, 4), -1, 1)
        step_result = env.step(act_random)
        for agent_idx in range(20):
            experience = Experience(state[agent_idx, :],
                                    act_random[agent_idx, :],
                                    step_result.rewards[agent_idx],
                                    step_result.next_state[agent_idx, :],
                                    step_result.done[agent_idx])
            agent.record_experience(experience)
        if np.any(step_result.done):
            break
    agent.experiences.sample()


random_test_run()
