from dec_envs.multiagentenv import MultiAgentEnv
from gym import Env
from gym.spaces import Discrete
import numpy as np


class DecPOMDPWrapper(Env):
    def __init__(self, env: MultiAgentEnv, action2actions, actions2action):
        self.env = env
        self.action_space:Discrete = Discrete(n=self.env.get_total_actions()**self.env.n_agents)
        self.observation_space:Discrete = Discrete(n=self.env.get_obs_size()*self.env.n_agents)
        self.action2actions = action2actions
        self.actions2action = actions2action

    def step(self, action):
        
        actions = self.action2actions[action]
        reward, done, _ = self.env.step(actions)
        next_obs = self.env.get_obs()[0].flatten()
        return next_obs, reward, done, {}, {}

    def reset(self):
        o, _ = self.env.reset()
        return o[0].flatten()

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def get_env_info(self):
        return self.env.get_env_info()
    