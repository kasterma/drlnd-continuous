# DRLND: Continuous Control Project

In this project the goal is to learn how to operate a double-jointed are to move to target locations; see the
environment section below for more details.

To start working with this project, clone the git repo and run

    make setup
    
in the root.  A `files` directory will be created with the required dependencies in it, together with a `venv` directory
with the virtual environment to be use (based on python 3.6).

A training run can be started using

    python train.py train --run_id=<NUM>
    
with <NUM> filled in with a numeric run identifier.  Result will be saved trained model files and a saved scores
file.  Then to evaluate how well the model really performs run

    python train.py evaluation --run_id=<NUM>
    
and then 100 episodes of interaction with the environment are run but without adding noise.  This gives a more honest
evaluation of how well the actor model really performs.  The scores of this run are also saved.

We have added the models and scores of a successful run in the `data` directory.

# Environment

# File organisation

In the `data` dir the results of the successful training run are stored (the models and both the training and evaluation
scores).

The `drlnd_continious` directory contains the key code other than the training driver.

The `images` directory contains the images for the report.

The `tests` directory contains some testing code to experiement with interacting with different parts of the environment
and other parts of the code.

The `venv` and `files` directories are created on `make setup` and contain the virtual env with all dependencies
installed and the different downloaded files (e.g. Reacher.app).

The `Makefile` contains all the commands needed to setup the environment

Finally the `train.py` file is the driver of the learning; this is where the training loop is located, where we run
the environment and pass the experiences to the agent to learn from.
