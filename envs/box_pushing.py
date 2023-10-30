from dec_envs.dec_box_pushing import DecBoxPushing
from envs.decpomdp2pomdp import DecPOMDPWrapper

# our range is 15 actions per agent

actions = [[i,j] for i in range(15) for j in range(15)]

action2actions = {
    k: actions for k, actions in enumerate(actions)
}

actions2action = {
    tuple(actions): k for k, actions in enumerate(actions)
}

class BoxPushing(DecPOMDPWrapper):
    
    def __init__(self):
        self.env = DecBoxPushing()
        super().__init__(self.env, action2actions, actions2action)

