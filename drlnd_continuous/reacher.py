# Reacher
#
# This class enables all interaction with the unity reacher environment.


import logging.config
from typing import List

import numpy as np
import yaml
from unityagents import UnityEnvironment

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("environment")


class StepResult:
    """Wrapper object for the information extracted from the unity environment"""

    def __init__(self, done: bool, rewards: List, new_state: np.ndarray):
        # Note: we check some types here, not very pythonic, but mostly to check understanding of the environment
        # This understanding will help avoid errors later on when using this data.
        assert type(done) == np.bool_
        assert type(rewards) == list and len(rewards) == 20
        assert new_state.shape == (20, 33)
        self.done = done
        self.rewards = rewards
        self.new_state = new_state


class Reacher:
    def __init__(self):
        self.env = UnityEnvironment(file_name="files/Reacher.app")
        log.info("Reacher environment set up")

    def reset(self, train_mode=True) -> None:
        self.env.reset(train_mode=train_mode)
        log.info("Reacher environment reset with train_mode=%s", train_mode)

    def step(self, action: np.ndarray) -> StepResult:
        """Take the action in the environment and collect the relevant information to return from the environment"""
        log.debug("taking step")
        assert action.shape == (20, 4)
        action_clipped = np.clip(action, -1, 1)
        env_info = self.env.step(action_clipped)["ReacherBrain"]
        return StepResult(done=np.any(env_info.local_done),
                          rewards=env_info.rewards,
                          new_state=env_info.vector_observations)
