from dec_envs.multiagentenv import MultiAgentEnv
import numpy as np

class DecBoxPushing(MultiAgentEnv):
    def __init__(self, width, height, n_agents = 2, small_boxes = 1, big_boxes = 1):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.small_boxes = small_boxes
        self.big_boxes = big_boxes

        self.agents_positions = [self.init_random_location() for _ in range(n_agents)] # Initial agents positions
        self.small_box_positions = [self.init_random_location(isbox = True) for _ in range(small_boxes)] # Initial small boxes positions
        self.big_box_positions = [self.init_random_location(isbox = True) for _ in range(big_boxes)] # Initial big boxes positions
        self.starting_big_positions = self.big_box_positions.copy()
        self.starting_small_positions = self.small_box_positions.copy()
        self.goal_position = (self.width - 1, self.height - 1) # Goal position
        self.observations = [[[0 for _ in range(self.small_boxes)],[0 for _ in range(self.big_boxes)]] for _ in range(n_agents)] # Observations


    def init_random_location(self, isbox = False):
        x = np.random.randint(0, self.width - 1)
        y = np.random.randint(0, self.height - 1)
        if isbox and (x, y) == (self.width - 1, self.height - 1):
            return self.init_random_location(isbox = True)
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
        

    def push_box(self, direction, box_id, isBig = False):
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

    def sense(self, box_id, agent_id, isBig = False):
        if isBig:
          box_x, box_y = self.big_box_positions[box_id]
        else:
          box_x, box_y = self.small_box_positions[box_id]
        agent_x, agent_y = self.agents_positions[agent_id]
        if box_x == agent_x and box_y == agent_y:
            return True
        return False
    
    def execute_actions(self, actions):
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
        
        elif action[0] == "move": # action[1] = direction
            self.move_agent(action[1], agent_id)

        elif action[0] == "sense": # action[1] = box_id, action[2] = isBig
            self.observations[agent_id][action[2]][action[1]] = self.sense(action[1], agent_id, action[2])

        elif action[0] == "push": # action[1] = direction, action[2] = isBig, action[3] = box_id
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
        self.move_box(box_id, move_small_left[box_id], move_small_right[box_id], move_small_up[box_id], move_small_down[box_id])
      for box_id in range(self.big_boxes):
        self.move_box(box_id, move_big_left[box_id], move_big_right[box_id], move_big_up[box_id], move_big_down[box_id], isBig = True)

    def move_box(self, box_id, left, right, up, down, isBig = False):
      direction = None
      thereshold = 2 if isBig else 1
      if left >= thereshold:
        if right + up + down == 0:
          direction = "left"
        else:
           return
      if right >= thereshold:
        if left + up + down == 0:
          direction = "right"
        else:
           return
      if up >= thereshold:
        if left + right + down == 0:
          direction = "up"
        else:
           return
      if down >= thereshold:
        if left + right + up == 0:
          direction = "down"
        else:
           return
      if direction is None:
        return
      self.push_box(direction, box_id, isBig)

    def get_obs(self):
        old_observations = [self.observations[agent_id][0].append(self.observations[agent_id][1]) for agent_id in range(self.n_agents)]
        self.observations = [[[0 for _ in range(self.small_boxes)],[0 for _ in range(self.big_boxes)]] for _ in range(self.n_agents)]
        return old_observations
    
    def get_obs_agent(self, agent_id):
        old_observations = self.observations[agent_id][0].append(self.observations[agent_id][1])
        self.observations[agent_id] = [[0 for _ in range(self.small_boxes)],[0 for _ in range(self.big_boxes)]]
        return old_observations
    
    def reset(self):
        self.agents_positions = [self.init_random_location() for _ in range(self.n_agents)]
        self.small_box_positions = self.starting_small_positions.copy()
        self.big_box_positions = self.starting_big_positions.copy()
        return self.get_obs(), None
