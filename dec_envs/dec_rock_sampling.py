from dec_envs.multiagentenv import MultiAgentEnv
import math
import random

class DecRockSampling(MultiAgentEnv):
    def __init__(self, width=2, height=2, bad_rocks=1, good_rocks=1, horizon = 20):
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

        self.observations = None

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
            self.bad_rocks_positions.add((x, y))
            self.sampled_rocks += 1
        elif rock_quality == 'bad':
            return -500
        if sampled_rocks == self.good_rocks:
                return 750
        else:
            return 0

    def execute_actions(self, actions):
        total_reward = 0
        for action in actions:
            if action[0] == "no-op":
                continue

            elif action[0] == "move":  # action[1] = rover_id, action[2] = direction
                total_reward -= 1
                if action[1] == 0:
                    self.move_rover1(action[2])
                elif action[1] == 1:
                    self.move_rover2(action[2])

            elif action[0] == "sense":  # action[1] = rover_id
                total_reward -= 5
                if action[1] == 0:
                    self.observations.add([self.sense(self.rover1_position), rover1_position])
                elif action[1] == 1:
                    self.observations.add([self.sense(self.rover2_position), rover2_position])

            elif action[0] == "sample":  # action[1] = rover_id
                if action[1] == 0:
                    total_reward += self.sample(self.rover1_position)
                elif action[1] == 1:
                    total_reward += self.sample(self.rover2_position)

        return total_reward


    def step(self, actions):     
        self.step_count += 1
        if self.step_count >= self.horizon:
            reward = 0
            self.step_count = 0
            return reward, done
        
        actions = self.translate_actions(actions)
        reward = self.execute_actions(actions)
        return reward, done

    def translate_actions(self, actions):
        translated_actions = []
        noop = [0]
        move1 = [1, 2, 3, 4]
        move2 = [5, 6, 7, 8]

        for action in actions:
            if action in noop:
                translated_actions.append(("no-op",))
            elif action in move:
                if action == 1:
                    translated_actions.append(("move",0, "left"))
                elif action == 2:
                    translated_actions.append(("move", 0, "right"))
                elif action == 3:
                    translated_actions.append(("move", 0, "up"))
                elif action == 4:
                    translated_actions.append(("move", 0, "down"))

            
            elif action in sense_small:
                translated_actions.append(("sense", action - 5, False))
            elif action in sense_big:
                translated_actions.append(("sense", action - 5 - k, True))
            elif action in push_small:
                box_id = (action - 5 - k - K) // 4
                direction = (action - 5 - k - K) % 4

                if direction == 0:
                    translated_actions.append(("push", "left", False, box_id))
                elif direction == 1:
                    translated_actions.append(("push", "right", False, box_id))
                elif direction == 2:
                    translated_actions.append(("push", "up", False, box_id))
                elif direction == 3:
                    translated_actions.append(("push", "down", False, box_id))
            elif action in push_big:
                box_id = (action - 5 - k - K - 4 * k) // 4
                direction = (action - 5 - k - K - 4 * k) % 4

                if direction == 0:
                    translated_actions.append(("push", "left", True, box_id))
                elif direction == 1:
                    translated_actions.append(("push", "right", True, box_id))
                elif direction == 2:
                    translated_actions.append(("push", "up", True, box_id))
                elif direction == 3:
                    translated_actions.append(("push", "down", True, box_id))
            else:
                raise IndexError

        return translated_actions