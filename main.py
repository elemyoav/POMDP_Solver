import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the Dectiger and ADRQN classes
from envs.dectiger import Dectiger
from model.DRQN import DRQN

vocab_actions = [("open-left", "open-left"), ("open-right", "open-right"), ("listen", "listen"), ("open-left", "open-right"), ("open-right", "open-left"), ("open-left", "listen"), ("listen", "open-right"), ("listen", "open-left"), ("open-right", "listen")]
vocab_obs = [("None", "None"), ("hear-left", "hear-left"), ("hear-left", "hear-right"), ("hear-right", "hear-left"), ("hear-right", "hear-right")]

def index2action(index):
    return vocab_actions[index]

def action2index(action):
    return vocab_actions.index(action)

def index2obs(index):
    return vocab_obs[index]

def obs2index(obs):
    return vocab_obs.index(obs)

# Define hyperparameters
n_obs = len(vocab_obs) # Number of observations: None, hear-left, hear-right
n_actions = len(vocab_actions) # Number of actions: open-left, open-right, listen
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 500
max_episodes = 5000


# Initialize the Dectiger environment
env = Dectiger()

actions = []

# Initialize the ADRQN agent
drqn = DRQN(n_obs, n_actions)

# Initialize progress plot
def train(max_episodes, env, drqn):
    # Training loop
    total_rewards = []
    epsilon = epsilon_start

    for episode in tqdm(range(max_episodes)):

        if episode % 100 == 0 and episode != 0:
            test_reward = test(100, env, drqn)
            total_rewards.append(test_reward)

        env.reset()
        done = False
        total_reward = 0
        o = "None", "None"

        episodee = []
        while not done:
            # Get the action
            action = drqn.act(obs2index(o), epsilon)
            action = index2action(action)

            # Execute the action
            next_o, r, done = env.step(action)

            # Reset the hidden state and add to total reward
            if done:
                drqn.reset_hidden_state()
            total_reward += r
            # Store the experience in the replay buffer

            episodee.append((np.array([[[obs2index(o)]]]), np.array([[[obs2index(next_o)]]]), action2index(action), r, done))
            o = next_o

            # Update the network
            drqn.update()
        
        drqn.replay_buffer.add(episodee)
        
        # Print the results
        print("Episode: {}, Total Reward: {}, epsilon: {}".format(episode, total_reward, epsilon))

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
    # Plot the results

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(total_rewards)
    plt.savefig('./results/total_rewards.png')
    plt.show()


def test(max_episodes, env, drqn):
    total_rewards = []

    for episode in range(max_episodes):
        env.reset()
        done = False
        total_reward = 0
        o = "None", "None"

        while not done:
            # Get the action
            action = drqn.act(obs2index(o), 0)
            action = index2action(action)

            # Execute the action
            next_o, r, done = env.step(action)

            # Reset the hidden state and add to total reward
            if done:
                drqn.reset_hidden_state()
            total_reward += r
            o = next_o

            drqn.update()
        
        print("test episode: {}, total reward: {}".format(episode, total_reward))

        total_rewards.append(total_reward)
    
    avg_total_reward = np.mean(total_rewards)
    print("Average total reward: {}".format(avg_total_reward))
    return avg_total_reward


if __name__ == "__main__":
    train(max_episodes, env, drqn)