from envs.dec_tiger import DecTiger
from model.DRQN import Agent

def main():
    env = DecTiger(**{"env_args": {}})
    agent = Agent(env)
    agent.train(max_episodes=2000)

if __name__ == "__main__":
    main()