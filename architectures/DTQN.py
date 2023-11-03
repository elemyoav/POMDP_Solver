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
from utils.positional_encodings import positional_encoding

tf.keras.backend.set_floatx('float64')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to use only the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # In case of an error, fall back to CPU
        print(e)
else:
    print("No GPU available, using CPU.")

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
            key_dim=256,
            num_actions = self.action_dim,
            num_layers=2,
            dff=256
        )

    def create_model(self, input_shape, num_heads, key_dim, num_actions, dff, num_layers):
        # Input layer
        
        inputs = Input(shape=input_shape)

        # project the input at each time step to a number
        embedded_inputs1 = Dense(units=key_dim // num_heads, activation='relu')(inputs)
        embedded_inputs2 = Dense(units=key_dim)(embedded_inputs1)
        
        # Positional Encoding
        x = embedded_inputs2 + positional_encoding(input_shape[0], key_dim)

        for _ in range(num_layers):
   
        # Multi-Head Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x, x, use_causal_mask=True)
            
            # Normalize the attention output
            x = LayerNormalization()(x + attention_output)

            # send it to a feed forward network
            ffn1 = Dense(units=dff, activation='relu')(x)
            ffn2 = Dense(units=dff, activation=None)(ffn1)

            # Normalize the ffn output
            x = LayerNormalization()(x + ffn2)
        
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
        q_value = q_values[-1]
        return np.argmax(q_value)

    def train(self, states, targets):
        states = tf.convert_to_tensor(states)
        self.model.fit(states, targets, verbose=0)

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

            not_done_mask = 1.0 - tf.cast(done, dtype=tf.float64)

            #Calculate the Q-value target without the rewards for the done time steps
            q_value_target = rewards + not_done_mask * args.gamma * next_q_values

            #Convert actions to integers and use it to index the third dimension of targets
            actions = tf.cast(actions, dtype=tf.int32)

            #Create a mask for the updated elements
            mask = tf.one_hot(actions, depth=self.action_dim, dtype=tf.float32)

            #Update the 'targets' tensor with Q-value_target for the corresponding actions
            targets = tf.where(tf.math.equal(mask, 1), q_value_target[..., tf.newaxis], targets)

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

            rewards = np.zeros(args.time_steps)
            dones = np.ones(args.time_steps)
            actions = np.zeros(args.time_steps)
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