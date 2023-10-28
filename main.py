from envs.tiger import Tiger
from model.DRQN import Agent

def main():
    env = Tiger()
    agent = Agent(env)
    agent.train(max_episodes=1500)

if __name__ == "__main__":
    main()