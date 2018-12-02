import logging.config

import numpy as np
import torch
import random

import yaml
from torch import optim

from drlnd_continuous.experience import Experience, Experiences
from drlnd_continuous.model import Actor, Critic
from drlnd_continuous.noise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0 # L2 weight decay


class Agent:
    def __init__(self, replay_memory_size, state_size, action_size, numpy_seed=36, random_seed=21, torch_seed=42):
        log.info("Random seeds, numpy %d, random %d, torch %d.", numpy_seed, random_seed, torch_seed)
        # seed all sources of randomness
        torch.manual_seed(torch_seed)
        # noinspection PyUnresolvedReferences
        np.random.seed(seed=numpy_seed)
        random.seed(random_seed)

        self.experiences = Experiences(memory_size=replay_memory_size, batch_size=BATCH_SIZE)

        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = self.actor_local.get_copy()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = self.critic_local.get_copy()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size)

    def record_experience(self, experience: Experience):
        self.experiences.add(experience)
        if True: # time to learn
            self.__learn()

    def get_action(self, state: np.ndarray, eps: float) -> np.ndarray:
        return self.model.eval(state)

    def save(self, run_identifier: str) -> None:
        pass

    def load(self, run_identifier: str) -> None:
        pass

    def __learn(self):
        pass
