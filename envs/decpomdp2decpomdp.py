
from dec_envs.multiagentenv import MultiAgentEnv
import numpy as np

def decpomdp2decpomdp(env_class:MultiAgentEnv, actions2action, action2actions)->MultiAgentEnv:
    """
    gets an n agents dec-pomdp environment and returns a 1 agent dec-pomdp environment
    """

    class ShadowClass(env_class):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        # use the built in reset

        #use the built in 

        def get_obs_agent(self, agent_id):
            assert agent_id == 0
            return self.get_obs()
        
        def get_avail_agent_actions(self, agent_id):
            assert agent_id == 0
            return actions2action(super().get_avail_actions())
        
        def get_avail_actions():
            return action2actions(super().get_avail_actions())
        

