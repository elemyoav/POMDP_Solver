import random
import numpy as np

class Dectiger:
    def __init__(self):
        self.S = ["tiger-left", "tiger-right"]
        self.A = ["open-left", "open-right", "listen"]
        self.numAgents = 2
        self.s = None
        self.horizon = 30
        self.currentStep = 0

        self.action_space = [("open-left", "open-left"), ("open-right", "open-right"), ("listen", "listen"), ("open-left", "open-right"), ("open-right", "open-left"), ("open-left", "listen"), ("listen", "open-right"), ("listen", "open-left"), ("open-right", "listen")]
        self.observation_space = [("None", "None"), ("hear-left", "hear-left"), ("hear-left", "hear-right"), ("hear-right", "hear-left"), ("hear-right", "hear-right")]
        self.n_obs = len(self.observation_space)

    def reset(self):
        self.s = random.choice(self.S)
        return self.one_hot(("None", "None"))

    def one_hot(self, obs):
        one_hot = np.zeros(self.n_obs)
        one_hot[self.observation_space.index(obs)] = 1
        return one_hot
    
    def step(self, actions):
        self.currentStep += 1
        actions = self.action_space[actions]
        if self.done(actions):
            self.currentStep = 0
            return self.one_hot(self.O(actions)), self.R(actions), True
        if self.currentStep >= self.horizon:
            self.currentStep = 0
            return self.one_hot(self.O(actions)), -200, True
        return self.one_hot(self.O(actions)), self.R(actions), False

    def O(self, actions):
        if all([action == "listen" for action in actions]):
            if self.s == self.S[0]:
                dice = random.random()
                if dice < 0.7225:
                    return "hear-left", "hear-left"
                elif dice < 0.85:
                    return "hear-left", "hear-right"
                elif dice < 0.9775:
                    return "hear-right", "hear-left"
                else:
                    return "hear-right", "hear-right"
            else:
                dice = random.random()
                if dice < 0.7225:
                    return "hear-right", "hear-right"
                elif dice < 0.85:
                    return "hear-left", "hear-right"
                elif dice < 0.9775:
                    return "hear-right", "hear-left"
                else:
                    return "hear-left", "hear-left"
        else:
            return "None", "None"

    def R(self, actions):
        a1, a2 = tuple(actions)
        if a1 == "listen" and a2 == "listen":
            return -2
        elif a1 == "open-left" and a2 == "open-left":
            if self.s == "tiger-right":
                return 20
            else:
                return -50
        elif a1 == "open-right" and a2 == "open-right":
            if self.s == "tiger-left":
                return 20
            else:
                return -50
        elif a1 == "open-left" and a2 == "open-right":
            return -100
        elif a1 == "open-right" and a2 == "open-left":
            return -100
        elif a1 == "open-left" and a2 == "listen":
            if self.s == "tiger-left":
                return -101
            else:
                return 9
        elif a1 == "listen" and a2 == "open-right":
            if self.s == "tiger-right":
                return -101
            else:
                return 9
        elif a1 == "listen" and a2 == "open-left":
            if self.s == "tiger-left":
                return -101
            else:
                return 9
        elif a1 == "open-right" and a2 == "listen":
            if self.s == "tiger-right":
                return -101
            else:
                return 9

    def done(self, actions):
        return not all([a == 'listen' for a in actions])