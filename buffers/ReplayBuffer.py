from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, args, capacity=10000):
        self.args = args
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, self.args.time_steps, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, self.args.time_steps, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)
