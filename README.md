# Deep Reinforcement Learning Project 1
This is an implementation of Deep Q-Learning for the first project in Udacity's Deep Reinforcement Learning class.  The model is built and trained to solve Unity's [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) scenario.


## Environment Details
The Banana Collector learning environment has the following properties:

  * State Space - The state space has 37 dimensions, including the agent's velocity and perception of nearby objects.
  * Action Space - There are four discrete actions available: move forward, move backward, turn left, and turn right.
  * Goal - The Unity Github page specifies a goal of 10.  However for this project my goal was to achieve a mean of 13 points over per episode.

For more information see the [Unity Github repo](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector)

## Agent Details
The agent uses Deep Q-Learning with experiance replay and fixed q-targets.  It is implemented in python3 using PyTorch, and uses a two hidden layers.

## Installation Requirements
  1. Create a python 3.6 virtual environment.  I used Anaconda for this.
  2. After activating the environment, pip install the requirements file.

## Running
The main.py file can be run with any of the following flags.

* `--examine`: Prints information on the learning environment.
* `--random`: Runs an agent that takes random actions.
* `--train`: Trains a new agent including saving a checkpoint of the model weights and printing out scores.
* `--test`: Runs an agent using the included checkpoint file.
