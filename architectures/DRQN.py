import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from buffers.ReplayBuffer import ReplayBuffer
import numpy as np
import random
from tqdm import tqdm
from envs.decpomdp2pomdp import DecPOMDPWrapper
from args import args

tf.keras.backend.set_floatx('float64')


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.opt = Adam(args.lr)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential([
            Input((args.time_steps, self.state_dim)),
            LSTM(32),
            Dense(32, activation='relu'),
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
    def __init__(self, env: DecPOMDPWrapper, N=15):
        self.env: DecPOMDPWrapper = env
        self.state_dim = env.observation_space.n
        self.action_dim = self.env.action_space.n

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
                self.update_states(self.env.reset())
                while not done:
                    action = self.model.get_action(self.states)
                    next_obs, reward, done, _, _ = self.env.step(action)

                    prev_states = self.states
                    self.update_states(next_obs)
                    self.buffer.put(prev_states, action,
                                    reward, self.states, done)
                    total_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay()
                self.catch_up()
                print('EP{} EpisodeReward={} epsilon={}'.format(
                    ep, total_reward, self.model.epsilon))

        finally:
            self.plot_data(progression)

            c = input('Save model? [y/n]')
            if c == 'y':
                self.model.model.save(f'./models/DRQN_{args.env}.h5')


    def test(self, max_episodes=100):

        total_rewards = []
        epsilon = self.model.epsilon
        self.model.epsilon = 0
        #lets add to the for loop a testing.. tqdm format
        for ep in tqdm(range(max_episodes), desc="Testing..."):
            done, total_reward = False, 0
            self.states = np.zeros([args.time_steps, self.state_dim])
            self.update_states(self.env.reset())
            while not done:
                action = self.model.get_action(self.states)
                next_obs, reward, done, _, _ = self.env.step(action)

                prev_states = self.states
                self.update_states(next_obs)
                self.buffer.put(prev_states, action, reward, self.states, done)
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

    def plot_data(self, data):
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.plot(data)
            plt.savefig(f'./results/total_rewards_{args.env}_{args.model}.png')
            plt.show()