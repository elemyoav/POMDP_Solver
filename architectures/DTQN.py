import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from buffers.ReplayBuffer import ReplayBuffer
import numpy as np
import random
from tqdm import tqdm
from envs.decpomdp2pomdp import DecPOMDPWrapper
from args import args
from PositionalEmbedding import positional_encoding

tf.keras.backend.set_floatx('float64')


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.opt = Adam(args.lr)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.create_model(
            input_shape=(args.time_steps, self.state_dim),
            num_heads=8,
            key_dim=self.state_dim,
            num_actions = self.action_dim,
            num_layers=2
        )

    def create_model(self, input_shape, num_heads, key_dim, num_actions, num_layers=4):
        # Input layer
        
        inputs = Input(shape=input_shape)

        # project the input at each time step to a number
        embedded_inputs = Dense(units=key_dim, activation=None)(inputs)
        
        # Positional Encoding
        x = embedded_inputs + positional_encoding(input_shape[0], input_shape[-1])
        #x = inputs
        for _ in range(num_layers):
        # Multi-Head Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x, x, use_causal_mask=True)
            
            # Normalize the attention output
            x = LayerNormalization()(x + attention_output)

            # send it to a feed forward network
            #ffn1 = Dense(units=32, activation='relu')(x)
            #ffn2 = Dense(units=32, activation='tanh')(ffn1)
            ffn = Dense(units=input_shape[-1], activation='relu')(x)

            # Normalize the ffn output
            x = LayerNormalization()(x + ffn)
        
        # Output layer
        output = Dense(units=num_actions, activation=None)(x)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=self.opt)
        
        model.summary()
        return model

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action(self, state):
        state = np.reshape(state, [1, args.time_steps, self.state_dim])
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        q_values = self.predict(state)[0]
        q_value = q_values[0]
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
            next_q_values = self.target_model.predict(next_states).max(axis=2)

            for i in range(args.batch_size):
                for j in range(args.time_steps):
                    targets[i, j, int(actions[i,j])] = rewards[i,j] + (1 - done[i,j]) * args.gamma * next_q_values[i,j]

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
                
                rewards = np.zeros(args.time_steps)
                dones = np.ones(args.time_steps)
                actions = np.zeros(args.time_steps)

                done, total_reward = False, 0
                self.states = np.zeros([args.time_steps, self.state_dim])
                self.update_states(self.env.reset())
                while not done:
                    action = self.model.get_action(self.states)
                    next_obs, reward, done, _, _ = self.env.step(action)

                    rewards = np.roll(rewards, -1)
                    rewards[-1] = reward

                    dones = np.roll(dones, -1)
                    dones[-1] = done

                    actions = np.roll(actions, -1)
                    actions[-1] = action


                    prev_states = self.states
                    self.update_states(next_obs)
                    self.buffer.put(prev_states, actions,
                                    rewards, self.states, dones)
                    
                    total_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay()
                self.catch_up()
                print('EP{} EpisodeReward={} epsilon={}'.format(
                    ep, total_reward, self.model.epsilon))

        finally:
            self.plot_data(progression)
            
            c = input('continue_training? [y/n]')
            if c == 'y':
                c = input('How many more episodes?')
                max_episodes = int(c)
                self.train(max_episodes)
                return
            
            c = input('Save model? [y/n]')
            if c == 'y':
                self.model.model.save(f'./models/{args.model}_{args.env}')


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

            #    prev_states = self.states
            #    self.update_states(next_obs)
            #    self.buffer.put(prev_states, action, reward, self.states, done)
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