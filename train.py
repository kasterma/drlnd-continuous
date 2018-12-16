# Training driver for the DRLND continuous project
#
# This code is the primary interface to start and evaluate the training for the continous project.
import logging.config
from collections import deque

import yaml

from drlnd_continuous import *
import numpy as np

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")


def random_test_run():
    """Take 100 randomly generated steps in the environment

    In this interaction also interact with the Agent (to test this interaction while developing it)
    """
    env = Reacher()
    agent = Agent(10000, action_size=4, actor_count=20, state_size=33)
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


# random_test_run()


def train_run(number_episodes: int =500, print_every: int =1, run_id=0, scores_window=100):
    """Perfor a training run

    :param number_episodes the number of episodes to run through
    :param max_t max length of an episode
    :param print_every give an update on progress after this many episodes
    :param run_id id to use in saving models
    """
    log.info("Run with id %s", run_id)
    env = Reacher()
    agent = Agent(replay_memory_size=100000, state_size=33, action_size=4, actor_count=20)
    state = env.reset(train_mode=False)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    for episode_idx in range(number_episodes):
        env.reset()
        agent.reset()
        score = 0
        while True:
            # noinspection PyUnresolvedReferences
            action = agent.get_action(state)
            step_result = env.step(action)
            for agent_idx in range(20):
                experience = Experience(state[agent_idx, :],
                                        action[agent_idx, :],
                                        step_result.rewards[agent_idx],
                                        step_result.next_state[agent_idx, :],
                                        step_result.done[agent_idx])
                agent.record_experience(experience)
                score += step_result.rewards[agent_idx]
            if np.any(step_result.done):
                break
            state = step_result.next_state
        scores.append(score/20)
        scores_deque.append(score/20)
        if episode_idx % print_every == 0:
            log.info("%d Mean score over last %d episodes %f", episode_idx, scores_window, np.mean(scores_deque))
        if np.mean(scores_deque) > 30:
            log.info("train success")
            break
    agent.experiences.sample()
    log.info("Saving models under id %s", run_id)
    agent.save(run_id)
    log.info("Saving scores to file scores-%d.npy", run_id)
    np.save("scores-{}.npy".format(run_id), np.array(scores_deque))

# train_run(run_id=2)


def test_run(number_episodes: int = 100, print_every: int = 1, run_id=0, scores_window=100):
    log.info("Run test with id %s", run_id)
    env = Reacher()
    agent = Agent(replay_memory_size=100000, state_size=33, action_size=4, actor_count=20)
    agent.load(run_id)
    state = env.reset(train_mode=True)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    for episode_idx in range(number_episodes):
        env.reset(train_mode=True)
        score = 0
        ct = 0
        while True:
            ct += 1
            # noinspection PyUnresolvedReferences
            action = agent.get_action(state, add_noise=False)
            step_result = env.step(action)
            #print(step_result.rewards)
            score += np.mean(step_result.rewards)
            if np.any(step_result.done):
                break
            state = step_result.next_state
        scores.append(score)
        scores_deque.append(score)
        if episode_idx % print_every == 0:
            log.info("%d Mean score over last %d episodes %f (%d)", episode_idx, scores_window, np.mean(scores_deque), ct)

    np.save("evaluate-scores-{}.npy".format(run_id), np.array(scores_deque))


test_run(run_id=2)
