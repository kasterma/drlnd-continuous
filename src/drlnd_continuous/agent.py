import numpy as np
import torch

from src.drlnd_continuous.experience import Experience, Experiences

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, replay_memory_size):
        self.experiences = Experiences(memory_size=replay_memory_size, batch_size=10)
        self.model = None

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
