# Project 1 Report

## Algorithm
The learning algorithm used was Q-learning, including both experiance replay and fixed q-targets.

## Hyperparameters
By and large I didn't change the hyperparameters much from the values used in other projects and focused on model architecture instead.  The one exception was epsilon decay, which I set automatically depending on the number of episodes.

  * Epsilon start (eps_start): 1
  * Epsilon end (eps_end): 0.01
  * Epsilon decay (eps_decay): =(eps_end ** (1 / n_episodes)) / eps_start
  * Gamma: 0.99
  * Tau: 1e-3
  * Learning rate (lr): 5e-4
  * Replay Buffer size (buffer_size): int(1e5)
  * Batch size (batch_size): 64
  * Fixed Q update interval (update_every): 4
  * Number of episodes (n_episodes): 2000

## Model Architecture
The neural network architecture I settled on was simple, containing two hidden layers with 128 and 64 nodes respectively.  I chose this after trying many architectures, some of which I included in the model file.  In general I found that small neural networks performed better.  The larger and more complicated neural networks started over fitting after a point, and even their peak performance as not as good as the smaller networks.  The reqard comparisons can be found below.

## Rewards
Rewards for the chosen model, trained over 2000 episodes.
The agent required only about 1200 episodes to consistently perform better than the goal.

![Final Results](https://github.com/rbak/deep-reinforcement-learning-project-1/blob/master/results/final.png)

Reward summaries for all included models, trained over 2000 episodes.

![Results Summary](https://github.com/rbak/deep-reinforcement-learning-project-1/blob/master/results/summary.png)

## Future Improvements
There are several improvements I would like to test with this project.  The main improvement would be to implement other know improvments to the Q-learning algorithm, including double DQN,  dueling DQN or priority replay.  I hope to revist this project to see the effects of these improvements both individually, and working together, possibly working towards implements all of the improvements in Rainbow.  I would also like to spend more time exploring the hyperparameters to see how they affect both the learning curve and final result.
