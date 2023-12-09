from dec_envs.multiagentenv import MultiAgentEnv
import math
import random

class DecRockSampling(MultiAgentEnv):
    def __init__(self, width=2, height=2, bad_rocks=1, good_rocks=1):
        self.width = width
        self.height = height
        self.good_rocks = good_rocks
        self.bad_rocks = bad_rocks
        self.rover1_position = (0, 0)
        self.rover2_position = (0, self.width - 1)
        self.sampled_rocks = 0

        self.good_rocks_positions = [self.init_random_location() for _ in range(good_rocks)]
        self.bad_rocks_positions = [self.init_random_location() for _ in range(bad_rocks)] 

    def init_random_location(self):
        x = random.randint(0, self.width)
        y = random.randint(0, self.height)
        return (x, y)

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

    def sense(self, agent_position, sense_range):
        x, y = agent_position
        sensed_area = self.grid[x - sense_range : x + sense_range + 1]
        sensed_area = [row[y - sense_range : y + sense_range + 1] for row in sensed_area]
        return sensed_area


    def sample(self, agent_position):
        x, y = agent_position
        rock_quality = self.grid[x][y]
        if rock_quality == 'good':
            self.grid[x][y] = 'bad'
            self.good_rocks += 1
            return 750
        elif rock_quality == 'bad':
            return -500
        else:
            return 0