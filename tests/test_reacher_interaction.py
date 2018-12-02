# Reacher Interaction
#
# Learn to interact with the unity environment and check assumptions about that environment.

import logging.config
import yaml
import numpy as np

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("interact")

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="files/Reacher.app")


def test_brain_parameters():
    """Assert on all brain parameters

    Note: brain here is a unityagents.brain.BrainParameters object; really just a dict collecting some relevant params
    """
    assert len(env.brain_names) == 1
    assert 'ReacherBrain' == env.brain_names[0]
    brain = env.brains['ReacherBrain']
    assert 33 == brain.vector_observation_space_size
    assert 'continuous' == brain.vector_observation_space_type
    assert 4 == brain.vector_action_space_size
    assert 'continuous' == brain.vector_action_space_type
    assert brain.__dict__ == {'brain_name': 'ReacherBrain',
                              'vector_observation_space_size': 33,
                              'num_stacked_vector_observations': 1,
                              'number_visual_observations': 0,
                              'camera_resolutions': [],
                              'vector_action_space_size': 4,
                              'vector_action_descriptions': ['', '', '', ''],
                              'vector_action_space_type': 'continuous',
                              'vector_observation_space_type': 'continuous'}


def test_agent_parameters():
    """Assert on agent parameters"""
    env_info = env.reset(train_mode=True)

    env_info = env_info['ReacherBrain']
    assert type(env_info.vector_observations) is np.ndarray
    assert env_info.vector_observations.shape == (20, 33)
    assert len(env_info.agents) == 20


def run_only_one_arm(arm_idx=3, steps=10, action: np.ndarray =None):
    """Interact with only one arm and let the others have no action.

    Note: the arms are not numbered sequentially/orderly in the environment.

    >>> run_only_one_arm(10, 100, action=np.array([0,0,0,1]))

    The next shows that if you act and then stop, gravity takes over.
    >>> actions = np.zeros((100, 4))
    >>> actions[1:5, :] = 1
    >>> run_only_one_arm(10, 100, action=actions)

    The arm consists of two connected pendulum acting in a gravity field.
    """
    if action.ndim == 2:
        assert action.shape[1] == 4
    else:
        assert len(action) in {1, 4}
    # noinspection PyStatementEffect
    env.reset(train_mode=False)["ReacherBrain"]
    num_agents = 20
    action_size = 4
    if action.ndim == 2:
        steps = action.shape[0]
    if action.ndim == 1:
        action = np.broadcast_to(action, (steps, 4))
    for idx in range(steps):
        # noinspection PyUnresolvedReferences
        act_random = np.clip(np.random.randn(1, action_size), -1, 1)
        # noinspection PyUnresolvedReferences
        actions = np.zeros((num_agents, action_size))
        actions[arm_idx, :] = action[idx, :] if action is not None else act_random
        env_info = env.step(actions)["ReacherBrain"]
        if np.any(env_info.local_done):
            break
