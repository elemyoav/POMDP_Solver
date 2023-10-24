import random

class Dectiger:
    def __init__(self):
        self.S = ["tiger-left", "tiger-right"]
        self.A = ["open-left", "open-right", "listen"]
        self.numAgents = 2
        self.s = None
        self.horizon = 30
        self.currentStep = 0

    def reset(self):
        self.s = random.choice(self.S)

    def step(self, actions):
        self.currentStep += 1
        if self.currentStep >= self.horizon or self.done(actions):
            self.currentStep = 0
            return self.O(actions), self.R(actions), True
        return self.O(actions), self.R(actions), False

    def O(self, actions):
        if all([action == "listen" for action in actions]):
            if self.s == "tiger-left":
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
