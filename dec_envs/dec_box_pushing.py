from dec_envs.multiagentenv import MultiAgentEnv
import numpy as np


class DecBoxPushing(MultiAgentEnv):
    def __init__(self, width=2, height=2, n_agents=2, small_boxes=1, big_boxes=1, horizon = 20):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.small_boxes = small_boxes
        self.big_boxes = big_boxes
        self.horizon = horizon
        self.step_count = 0

        self.agents_positions = [self.init_random_location(
        ) for _ in range(n_agents)]  # Initial agents positions
        self.small_box_positions = [self.init_random_location(
            isbox=True) for _ in range(small_boxes)]  # Initial small boxes positions
        self.big_box_positions = [self.init_random_location(
            isbox=True) for _ in range(big_boxes)]  # Initial big boxes positions
        self.starting_big_positions = self.big_box_positions.copy()
        self.starting_small_positions = self.small_box_positions.copy()
        self.goal_position = (self.width - 1, self.height - 1)  # Goal position

        self.observations = None
        self.reset_observations()

        self.small_reach = [False for _ in range(small_boxes)]
        self.big_reach = [False for _ in range(big_boxes)]

    def init_random_location(self, isbox=False):
        x = np.random.randint(0, self.width)
        y = np.random.randint(0, self.height)
        if isbox and (x, y) == (self.width - 1, self.height - 1):
            return self.init_random_location(isbox=True)
        return (x, y)

    def move_agent(self, direction, agent_id):
        x, y = self.agents_positions[agent_id]
        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < self.width - 1:
            x += 1

        self.agents_positions[agent_id] = (x, y)

    def push_box(self, direction, box_id, isBig=False):
        if isBig:
            x, y = self.big_box_positions[box_id]
        else:
            x, y = self.small_box_positions[box_id]

        if direction == 'down' and y > 0:
            y -= 1
        elif direction == 'up' and y < self.height - 1:
            y += 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < self.width - 1:
            x += 1

        if isBig:
            self.big_box_positions[box_id] = (x, y)
        else:
            self.small_box_positions[box_id] = (x, y)

    def sense(self, box_id, agent_id, isBig=False):
        if isBig:
            box_x, box_y = self.big_box_positions[box_id]
        else:
            box_x, box_y = self.small_box_positions[box_id]
        agent_x, agent_y = self.agents_positions[agent_id]
        if box_x == agent_x and box_y == agent_y:
            return True
        return False

    def execute_actions(self, actions):
        total_reward = 0
        move_big_left = [0 for _ in range(self.big_boxes)]
        move_big_right = [0 for _ in range(self.big_boxes)]
        move_big_up = [0 for _ in range(self.big_boxes)]
        move_big_down = [0 for _ in range(self.big_boxes)]
        move_small_left = [0 for _ in range(self.small_boxes)]
        move_small_right = [0 for _ in range(self.small_boxes)]
        move_small_up = [0 for _ in range(self.small_boxes)]
        move_small_down = [0 for _ in range(self.small_boxes)]

        for agent_id, action in enumerate(actions):
            if action[0] == "no-op":
                continue

            elif action[0] == "move":  # action[1] = direction
                total_reward -= 10
                self.move_agent(action[1], agent_id)

            elif action[0] == "sense":  # action[1] = box_id, action[2] = isBig
                total_reward -= 1
                self.observations[agent_id][action[2]][action[1]] = self.sense(
                    action[1], agent_id, action[2])

            elif action[0] == "push":  # action[1] = direction, action[2] = isBig, action[3] = box_id
                total_reward -= 30
                if not self.sense(action[3], agent_id, action[2]):
                    continue
                if action[2]:
                    if action[1] == "left":
                        move_big_left[action[3]] += 1
                    elif action[1] == "right":
                        move_big_right[action[3]] += 1
                    elif action[1] == "up":
                        move_big_up[action[3]] += 1
                    elif action[1] == "down":
                        move_big_down[action[3]] += 1
                else:
                    if action[1] == "left":
                        move_small_left[action[3]] += 1
                    elif action[1] == "right":
                        move_small_right[action[3]] += 1
                    elif action[1] == "up":
                        move_small_up[action[3]] += 1
                    elif action[1] == "down":
                        move_small_down[action[3]] += 1

        for box_id in range(self.small_boxes):
            total_reward += self.move_box(
                box_id, move_small_left[box_id], move_small_right[box_id], move_small_up[box_id], move_small_down[box_id])

        for box_id in range(self.big_boxes):
            total_reward += self.move_box(box_id, move_big_left[box_id], move_big_right[box_id],
                                          move_big_up[box_id], move_big_down[box_id], isBig=True)

        return total_reward

    def move_box(self, box_id, left, right, up, down, isBig=False):
        direction = None
        thereshold = 2 if isBig else 1
        if left >= thereshold:
            if right + up + down == 0:
                direction = "left"
            else:
                return 0
        if right >= thereshold:
            if left + up + down == 0:
                direction = "right"
            else:
                return 0
        if up >= thereshold:
            if left + right + down == 0:
                direction = "up"
            else:
                return 0
        if down >= thereshold:
            if left + right + up == 0:
                direction = "down"
            else:
                return 0
        if direction is None:
            return 0
        self.push_box(direction, box_id, isBig)

        if isBig:
            box_x, box_y = self.big_box_positions[box_id]
            if box_x == self.goal_position[0] and box_y == self.goal_position[1] and not self.big_reach[box_id]:
                self.big_reach[box_id] = True
                return 1000
            return 0
        else:
            box_x, box_y = self.small_box_positions[box_id]
            if box_x == self.goal_position[0] and box_y == self.goal_position[1] and not self.small_reach[box_id]:
                self.small_reach[box_id] = True
                return 500
            return 0

    def get_obs(self):
        old_observations = np.array([np.append(np.append(self.observations[i][0], self.observations[i][1]), self.get_agent_position(i)) for i in range(self.n_agents)])
        self.reset_observations()
        return [old_observations]

    def get_agent_position(self, agent_id):
        return np.array([self.agents_positions[agent_id][0], self.agents_positions[agent_id][1]])
    
    def reset_observations(self):
        self.observations = [[np.zeros(self.small_boxes), [np.zeros(self.big_boxes)]] for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        old_observation = np.append(self.observations[agent_id][0], self.observations[agent_id][1])
        self.observations[agent_id] = [np.zeros(self.small_boxes), [np.zeros(self.big_boxes)]]
        return old_observation

    def reset(self):
        self.agents_positions = [self.init_random_location()
                                 for _ in range(self.n_agents)]
        self.small_box_positions = self.starting_small_positions.copy()
        self.big_box_positions = self.starting_big_positions.copy()
        self.small_reach = [False for _ in range(self.small_boxes)]
        self.big_reach = [False for _ in range(self.big_boxes)]
        return self.get_obs(), None

    def step(self, actions):
        
        self.step_count += 1
        if self.step_count >= self.horizon:
            done = True
            reward = 0
            self.step_count = 0
            return reward, done, None
        
        actions = self.translate_actions(actions)
        reward = self.execute_actions(actions)
        done = self.check_done()
        return reward, done, None

    def get_total_actions(self):
        return 1 + \
            4 + \
            (self.small_boxes + self.big_boxes) + \
            (4 * self.small_boxes + 4 * self.big_boxes)

    def get_obs_size(self):
        return self.small_boxes + self.big_boxes + 2

    def check_done(self):
        return all(self.small_reach) and all(self.big_reach)

    def translate_actions(self, actions):
        """

        assume there are k small boxes and K big boxes
        0: no-op
        1: move left
        2: move right
        3: move up
        4: move down

        5: sense small box 0
        .
        .
        .
         5 + k - 1: sense small box k - 1

         5 + k: sense big box 0
         .
         .
         .
         5 + k + K - 1: sense big box K - 1

         5 + k + K: push small box 0 left
         5 + k + K + 1: push small box 0 right
         5 + k + K + 2: push small box 0 up
         5 + k + K + 3: push small box 0 down
         .
         .
         .
         5 + k + K + 4 * k - 1: push small box k - 1 down

         5 + k + K + 4 * k: push big box 0 left
         5 + k + K + 4 * k + 1: push big box 0 right
         5 + k + K + 4 * k + 2: push big box 0 up
         5 + k + K + 4 * k + 3: push big box 0 down
         .
         .
         .
         5 + k + K + 4 * k + 4 * K - 1: push big box K - 1 down
        """

        translated_actions = []
        k = len(self.small_box_positions)
        K = len(self.big_box_positions)

        noop = [0]
        move = [1, 2, 3, 4]
        sense_small = [5 + i for i in range(k)]
        sense_big = [5 + k + i for i in range(K)]
        push_small = [5 + k + K + 4 * i +
                      j for i in range(k) for j in range(4)]
        push_big = [5 + k + K + 4 * k + 4 * i +
                    j for i in range(K) for j in range(4)]

        for action in actions:
            if action in noop:
                translated_actions.append(("no-op",))
            elif action in move:
                if action == 1:
                    translated_actions.append(("move", "left"))
                elif action == 2:
                    translated_actions.append(("move", "right"))
                elif action == 3:
                    translated_actions.append(("move", "up"))
                elif action == 4:
                    translated_actions.append(("move", "down"))
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
