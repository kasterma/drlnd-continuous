from typing import List, Tuple

import numpy as np
from collections import deque
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NotEnoughExperiences(Exception):
    """Exception to throw when Experiences is asked for a sample but doesn't have enough data yet"""
    pass


class Experience:
    """Wrapper class for an experience as received from the environment"""

    def __init__(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        assert state.shape == (33,)
        self.state = state
        assert action.shape == (4,)
        self.action = action
        self.reward = reward
        assert next_state.shape == (33,)
        self.next_state = next_state
        self.done = done

    def __repr__(self):
        return "Experience(" + str(self.state) + "," + str(self.action) + "," + str(self.reward) + "," +\
               str(self.next_state) + "," + str(self.done) + ")"


class Experiences:
    """
    Fixed-size buffer to store experience tuples.

    Store the data as received from the Unity environment, but before returning the sample prepare it for use by the
    agent for learning (the agent can ignore that it seems to store numpy data and needs torch tensors).
    """

    def __init__(self, memory_size=None, batch_size=None, *, config=None):
        if config:
            assert memory_size is None
            assert batch_size is None
            memory_size = config['experience_memory']['size']
            batch_size = config['train']['batch_size']
        self.memory = deque(maxlen=memory_size)
        self.sample_size = batch_size      # in the current learning method we return a sample the size of a batch

    def add(self, experience: Experience) -> None:
        self.memory.append(experience)

    def sample(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        try:
            sample = random.sample(self.memory, k=self.sample_size)
        except ValueError:
            raise NotEnoughExperiences()

        def to_float_tensor(dat):
            return torch.from_numpy(np.vstack(dat)).float().to(device)

        states_tensor = to_float_tensor([e.state for e in sample])
        actions_tensor = to_float_tensor([e.action for e in sample])
        rewards_tensor = to_float_tensor([e.reward for e in sample])
        next_states_tensor = to_float_tensor([e.next_state for e in sample])
        dones_tensor = to_float_tensor(np.array([e.done for e in sample], dtype=np.uint8))   # need to preprocess the booleans

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

