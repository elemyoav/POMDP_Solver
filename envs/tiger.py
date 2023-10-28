from dec_envs.dec_tiger import DecTiger
from envs.decpomdp2pomdp import DecPOMDPWrapper


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


class Tiger(DecPOMDPWrapper):
    
    def __init__(self):

        env = DecTiger(**{"env_args": {}})
        super().__init__(env, ACTION2ACIONS, ACTIONS2ACTION)