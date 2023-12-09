from dec_envs.multiagentenv import MultiAgentEnv
import math

class DecRockSampling(MultiAgentEnv):
    def __init__(self, width=2, height=2, bad_rocks=1, good_rocks=1):
        self.width = width
        self.height = height
        self.good_rocks = good_rocks
        self.bad_rocks = bad_rocks
        self.rover1_position = (0, 0)
        self.rover2_position = (0, self.width - 1)
        self.sampled_rocks = 0


    def move_rover1(self, direction):
        x, y = self.rover1_position
        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < math.ceil(self.width / 2) - 1:
            x += 1
        self.rover1_position = (x, y)

    def move_rover2(self, direction):
        x, y = self.rover2_position
        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > math.ceil(self.width / 2) - 1:
            x -= 1
        elif direction == 'right' and x < self.width - 1:
            x += 1
        self.rover2_position = (x, y)