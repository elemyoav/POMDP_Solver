import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from RNN_Model import RNNModel

class DRQN:
    
    def __init__(self, n_obs, n_actions, learning_rate=5e-5, gamma=0.99, max_len=100000, batch_size=64, N=2000):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma

        self.model = self.create_model()
        self.old_model = self.create_model()
        self.old_model.set_weights(self.model.get_weights())

        self.replay_buffer = ReplayBuffer(max_len)
        self.batch_size = batch_size
        self.n = 0
        self.N = N
        self.hidden_state = None
    
    def create_model(self):
        model = RNNModel(self.n_actions)
        model.compile(loss='mse', optimizer=self.optimizer)

        return model
    
    def act(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            obs = np.array([[[obs]]])
            q_values, self.hidden_state = self.model(obs, initial_state=self.hidden_state)  
            action = np.argmax(q_values)  # Get the action with the highest Q-value
            return action
    
    def reset_hidden_state(self):
        self.hidden_state = None

    def catch_up(self):
        if self.n == self.N:
            self.n = 0
            self.old_model.set_weights(self.model.get_weights())
            print("Catching up...")
        else:
            self.n += 1

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of transitions from replay buffer
        episodes = self.replay_buffer.sample(self.batch_size)
        total_loss = 0

        for episode in episodes:
            old_network_state = None
            new_network_state = None

            for o, next_o, action, reward, done in episode:
                o = np.array([[[o]]])
                next_o = np.array([[[next_o]]])

                # Compute the target Q-values

                # Update the network
                with tf.GradientTape() as tape:
                    old_q_vals, old_network_state = self.old_model(o, initial_state=old_network_state)
                    max_q_val = np.max(old_q_vals)
                    new_q_vals, new_network_state = self.model(next_o, initial_state=new_network_state)

                    target = new_q_vals.numpy()
                    target[0][action] = reward + self.gamma * max_q_val * (1 - done)
                    target = tf.convert_to_tensor(target)

                    loss = tf.keras.losses.MSE(target, new_q_vals)
                    total_loss += loss

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                self.catch_up()

        total_loss /= self.batch_size
        print("Loss: {}".format(total_loss))
        


    
