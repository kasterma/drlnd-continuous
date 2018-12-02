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

# Note

We have set up the project in the form of a python package.  Final usage should be an install and then the
ability to run the commands.  Experience with some projects in R have shown that in the live of a project this
eventually pays off from better ability to run tests, and better resusabilty.  In this specific case I consider
it more of a practice to work with this package layout.

# TODO

- In the unity environment there is a parameter train_mode; currently unclear what that does to the environment.
