from gym import Env
from dec_envs.multiagentenv import MultiAgentEnv


class DecPomdpWrapper(MultiAgentEnv):

    def __init__(self, env:Env, action2actions, actions2action):
        self.env = env
        self.action2actions = action2actions
        self.actions2action = actions2action
        self.n_agents = 1
        self.current_obs = None
        self.current_state = None

    def step(self, actions):
        action = self.actions2action(actions)
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        return reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        self.current_obs = obs
        return obs, self.current_state
    
    def get_obs(self):
        return self.current_obs
    
    def get_state(self):
        return self.current_state

    def get_obs_agent(self, agent_id):
        assert agent_id == 0
        return self.current_obs

    def get_obs_size(self):
        return self.env.observation_space.shape[0]

    def get_state_size(self):
        return self.env.state_space.shape[0]
    
    def get_avail_actions(self):
        return [1 for _ in range(self.get_total_actions())]
    
    def get_avail_agent_actions(self, agent_id):
        assert agent_id == 0
        return self.get_avail_actions()
    
    def get_total_actions(self):
        return len(self.action2actions)
    
    def get_stats(self):
        return {}
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        self.env.seed()

    
