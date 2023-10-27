import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from buffers.ReplayBuffer import ReplayBuffer
import argparse
import numpy as np
import random
from tqdm import tqdm
from envs.multiagentenv import MultiAgentEnv

tf.keras.backend.set_floatx('float64')


ACTION2ACIONS = {
    0: [0, 0],
    1: [1, 0],
    2: [2, 0],
    3: [0, 1],
    4: [1, 1],
    5: [2, 1],
    6: [0, 2],
    7: [1, 2],
    8: [2, 2]
}


ACTIONS2ACTION = {
    (0, 0): 0,
    (1, 0): 1,
    (2, 0): 2,
    (0, 1): 3,
    (1, 1): 4,
    (2, 1): 5,
    (0, 2): 6,
    (1, 2): 7,
    (2, 2): 8
}


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--time_steps', type=int, default=30)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.opt = RMSprop(args.lr)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential([
            Input((args.time_steps, self.state_dim)),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action(self, state):
        state = np.reshape(state, [1, args.time_steps, self.state_dim])
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            assert targets.shape == logits.shape
            loss = self.compute_loss(targets, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def decay_epsilon(self):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)


class Agent:
    def __init__(self, env: MultiAgentEnv, N=15):
        self.env: MultiAgentEnv = env
        self.state_dim = env.get_obs_size()**self.env.n_agents
        self.action_dim = self.env.get_total_actions() ** self.env.n_agents

        self.states = np.zeros([args.time_steps, self.state_dim])

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer(args)

        self.n = 0
        self.N = N

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self, epochs=10):
        self.model.decay_epsilon()

        for _ in range(epochs):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + \
                (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def train(self, max_episodes=10000):
        try:
            progression = []
            for ep in tqdm(range(max_episodes)):

                if ep % 100 == 0:
                    avg = self.test()
                    progression.append(avg)

                done, total_reward = False, 0
                self.states = np.zeros([args.time_steps, self.state_dim])
                o = self.env.reset()[0][0].flatten()
                self.update_states(o)
                while not done:
                    action = self.model.get_action(self.states)
                    actions = self.action2actions(action)
                    reward, done, _ = self.env.step(actions)
                    next_state = self.env.get_obs()[0].flatten()

                    prev_states = self.states
                    self.update_states(next_state)
                    self.buffer.put(prev_states, action,
                                    reward*0.01, self.states, done)
                    total_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay()
                self.catch_up()
                print('EP{} EpisodeReward={} epsilon={}'.format(
                    ep, total_reward, self.model.epsilon))

        finally:
            self.plot_data(progression)


    def test(self, max_episodes=100):
        total_rewards = []
        epsilon = self.model.epsilon
        self.model.epsilon = 0
        for ep in range(max_episodes):
            done, total_reward = False, 0
            self.states = np.zeros([args.time_steps, self.state_dim])
            o = self.env.reset()[0][0].flatten()
            self.update_states(o)
            while not done:
                action = self.model.get_action(self.states)
                actions = self.action2actions(action)
                reward, done, _ = self.env.step(actions)
                next_state = self.env.get_obs()[0].flatten()

                prev_states = self.states
                self.update_states(next_state)
                self.buffer.put(prev_states, action, reward *
                                0.01, self.states, done)

                total_reward += reward
            total_rewards.append(total_reward)

        print('Test Average Reward: {}'.format(np.mean(total_rewards)))
        self.model.epsilon = epsilon
        return np.mean(total_rewards)

    def catch_up(self):
        if self.n == self.N:
            self.n = 0
            self.target_update()
            print("Catching up...")
        else:
            self.n += 1

    def action2actions(self, action):

        return ACTION2ACIONS[action]

    def actions2action(self, actions):
        return ACTIONS2ACTION[tuple(actions)]


    def plot_data(self, data):
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.plot(data)
            plt.savefig('./results/total_rewards.png')
            plt.show()