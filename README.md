# DRLND: Continuous Control Project

In this project the goal is to learn how to operate a double-jointed are to move to target locations.

To start working with this project, clone the git repo and run

    make setup
    
in the root.  A `files` directory will be created with the required dependencies in it, together with a venv directory
with the virtual environment to be use (based on python 3.6).

To start working with the environment run

    jupyter notebook
    
and open `files/Continous_Control.ipynb`.  In this notebook (as provided by udacity) you can see some initial
interaction with the Reacher environment, and with that see that the core dependencies are correctly installed.

# TODO

- In the unity environment there is a parameter train_mode; currently unclear what that does to the environment.
- Set up in the better style of a package with src/ directory and tox for testing
- Investiage the Ornstein-Uhlenbeck process some more; main question why not add some basic Gaussian noise
  - after have functioning training process, try with different noise distributions

# Notes

## run 0

Didn't learn.  Looked at the noise some more, and the values are off.  Fixed to
mean 0 noise for next run.

## run 1

Didn't learn.

## run 2

Trying to reduce the noise over time.
