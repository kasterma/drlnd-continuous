# Report: Deep Reinforcement Learning with Continuous Actions

In this project we are learning how to operate a robot arm with two joints (think shoulder and elbow).  The arm
functions as two pendulums attached together (if you stop applying force, gravity will pull the arm to point straigth
down; you can see this by applying some random actions followed by the all zero actions). 

![reacher](images/reacher.png)

In the environment you have a collection of these double-jointed robot arms whose hands (the blue spheres) should be
inside the targets the larger green spheres.  In this image there are 10 arms, but in the environment we interact with
there are 20.  Each of these arms is controlled through an action that consists of four numbers between -1 and 1
indicating the torque applied in two dimensions to each of the joints.  The observation per arm consists of 33 variables
corresponding to position, rotation, velocities, and angular velocities of the arm.  Any time step the robot hand is in
the target a reward of 0.04 is earned.

The environment is considered to be solved average episode score for 100 episodes is above 30.  The score of an
episode is the average over all 20 agents, and the score for an agent is the sum of its rewards (i.e. no discounting).
Since an episode consists of 1000 steps the maximum score that can be earned is 40.

## Learning Algorithm

We are using the DDPG, _d_eep _d_eterministic _p_olicy _g_radient, algorithm (from Continous Control with Deep
Reinforcement Learning, Lillicrap et al [archiv](https://arxiv.org/pdf/1509.02971.pdf)).  Our implementation is heavily
based on a provided reference implementation
[udacity pendulum reference implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)


### Hyperparameters

| Parameter    |       Value |
------------------------------
| BUFFER_SIZE  |  int(1e5)   |
| BATCH_SIZE   |  128        |
| GAMMA        |  0.99       |
| TAU          |  1e-3       |
| LR_ACTOR     |  1e-4       |
| LR_CRITIC    |  1e-4       |
| WEIGHT_DECAY |  0          |
| UPDATE_EVERY |  20         |

### Model Architectures

We have two models an Actor and a Critic model

![architecture](images/network_architecture.png)

## Training development

TODO: include plot of rewards per episode

![scores](images/both-scores.png)

## Future Work

The results obtained here were created using the example code as provided by udacity with minor tweaking of the
hyperparameters.  Since in the evaluation we are already close to perfect score (we achieve around 39 on all
episodes, and 40 is max achievable) there is just a small bit room for improvement in the final agent ability.  Also
from the plot of rewards per episode we see that after some 30 episodes the agent is essentially trained, hence there
is not much room for improvement in learning speed either.

Still using a more complicated challenge we would be very interested in implementing the distributed methods of learning
such as Distributed Distributional Deterministic Policy Gradients.

We expect to be able to get closer to optimal performance if allow the trained agent to continue training without noise
for some number of episodes.  Then it can refine its actions to be able to stay in the target.

On the other hand our understanding of the hyperparameters can use some more development, so we see the real future
work in analysing more what hyperparameter ranges work and how the different parameters interact.

Finally the effect of the noise process is not yet clear to us.  We want to run with different noise generation
(e.g. standard gaussian, and reducing the noise over time) and see the effect.
