from agent import Agent
import argparse
from collections import deque
from env import Environment
import numpy as np
import torch

from model import QNetwork, Small, Large, Dropout
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def main(args):
    if args.examine:
        examine()
    if args.random:
        random()
    if args.train:
        train_multiple(summary_interval=50)
    if args.test:
        test()


def examine():
    with Environment() as env:
        # reset the environment
        env_info = env.reset()

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))

        # number of actions
        action_size = env.action_space_size
        print('Number of actions:', action_size)

        # examine the state space
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)


def random():
    with Environment(no_graphics=False) as env:
        env_info = env.reset()
        action_size = env.action_space_size
        score = 0
        while True:
            action = np.random.randint(action_size)
            env_info = env.step(action)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break
            print('\rScore: {:.2f}'.format(score), end="")
        print('\rScore: {:.2f}'.format(score))


def test():
    with Environment(no_graphics=False) as env:
        env_info = env.reset(train_mode=False)
        action_size = env.action_space_size
        state_size = len(env_info.vector_observations[0])
        state = env_info.vector_observations[0]
        agent = Agent(model=QNetwork, state_size=state_size, action_size=action_size, seed=0)
        agent.qnetwork_local.load_state_dict(torch.load('results/checkpoint.pth'))
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break
            print('\rScore: {:.2f}'.format(score), end="")
        print('\rScore: {:.2f}'.format(score))


def train_multiple(summary_interval):
    models = [QNetwork, Small, Large, Dropout]
    with Environment() as env:
        summary_scores = {}
        for model in models:
            summary = train(model, env, summary_interval)
            summary_scores[model.__name__] = summary
        plot_summary(summary_scores, summary_interval)


def train(model, env, summary_interval=100):
    env_info = env.reset(train_mode=True)
    action_size = env.action_space_size
    state_size = len(env_info.vector_observations[0])
    agent = Agent(model=model, state_size=state_size, action_size=action_size, seed=0)
    model_name = model.__name__
    summary_scores = _train(env, agent, model_name=model_name, summary_interval=summary_interval)
    return summary_scores


def _train(env, agent, model_name='model', summary_interval=100,
           n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
    """
    scores = []                        # list containing scores from each episode
    summary_scores = []
    scores_window = deque(maxlen=summary_interval)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    eps_decay = (eps_end ** (1 / n_episodes)) / eps_start
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % summary_interval == 0:
            summary_scores.append(np.mean(scores_window))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/' + model_name + '.pth')
            plot_scores(scores, summary_scores, summary_interval, model_name)
        # if np.mean(scores_window) >= 20.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
        #         i_episode - summary_interval, np.mean(scores_window)))
        #     break
    torch.save(agent.qnetwork_local.state_dict(), 'results/' + model_name + '.pth')
    plot_scores(scores, summary_scores, summary_interval, model_name)
    return summary_scores


def plot_scores(scores, summary_scores, summary_interval, file_name=''):
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(summary_scores) * summary_interval), summary_scores)
    if file_name:
        plt.savefig('results' + file_name + '.png', bbox_inches='tight')
    else:
        plt.show()


def plot_summary(summary_scores, summary_interval):
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    for key, scores in summary_scores.items():
        plt.plot(np.arange(len(scores)) * summary_interval, scores, label=key)
    plt.legend()
    plt.savefig('results/summary.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepRL - Q-learning project')
    parser.add_argument('--examine',
                        action="store_true",
                        dest="examine",
                        help='Print environment information')
    parser.add_argument('--random',
                        action="store_true",
                        dest="random",
                        help='Start a random agent')
    parser.add_argument('--train',
                        action="store_true",
                        dest="train",
                        help='Train a new network')
    parser.add_argument('--test',
                        action="store_true",
                        dest="test",
                        help='Load an existing network and test it')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('No arguments provided.')
    main(args)
