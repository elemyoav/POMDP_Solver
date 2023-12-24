from dec_envs.multiagentenv import MultiAgentEnv
import math
import random

class DecRockSampling(MultiAgentEnv):
    def __init__(self, width=2, height=2, bad_rocks=1, good_rocks=1, horizon=20):
        self.width = width
        self.height = height
        self.good_rocks = good_rocks
        self.bad_rocks = bad_rocks
        self.rover1_position = (0, 0)
        self.rover2_position = (0, self.width - 1)
        self.sampled_rocks = 0
        self.step_count = 0
        self.horizon = horizon

        self.good_rocks_positions = [self.init_random_location() for _ in range(good_rocks)]
        self.bad_rocks_positions = [self.init_random_location() for _ in range(bad_rocks)]

        self.observations = []

    def init_random_location(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return (x, y)

    def move_rover1(self, direction):
        x, y = self.rover1_position
        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < math.floor(self.width / 2):
            x += 1
        self.rover1_position = (x, y)

    def move_rover2(self, direction):
        x, y = self.rover2_position
        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > math.floor(self.width / 2):
            x -= 1
        elif direction == 'right' and x < self.width - 1:
            x += 1
        self.rover2_position = (x, y)

    def sense(self, rover_position):
        x, y = rover_position
        if (x, y) in self.good_rocks_positions:
            return 'good'
        elif (x, y) in self.bad_rocks_positions:
            return 'bad'
        else:
            return 'none'

    def sample(self, rover_position):
        x, y = rover_position
        rock_quality = self.sense(rover_position)
        if rock_quality == 'good':
            self.good_rocks_positions.remove((x, y))
            self.bad_rocks_positions.append((x, y))
            self.sampled_rocks += 1
            if self.sampled_rocks == self.good_rocks:
                return 750
            return 0
        elif rock_quality == 'bad':
            return -500
        else:
            return 0

    def execute_actions(self, actions):
        total_reward = 0
        for action in actions:
            if action[0] == "no-op":
                continue
            elif action[0] == "move":
                total_reward -= 1
                if action[1] == 0:
                    self.move_rover1(action[2])
                elif action[1] == 1:
                    self.move_rover2(action[2])
            elif action[0] == "sense":
                total_reward -= 5
                if action[1] == 0:
                    self.observations.append([self.sense(self.rover1_position), self.rover1_position])
                elif action[1] == 1:
                    self.observations.append([self.sense(self.rover2_position), self.rover2_position])
            elif action[0] == "sample":
                if action[1] == 0:
                    total_reward += self.sample(self.rover1_position)
               
