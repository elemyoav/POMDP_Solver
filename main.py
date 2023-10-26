
from envs.dectiger import Dectiger
from model.DRQN import Agent

def main():
    env = Dectiger()
    agent = Agent(env)
    agent.train(max_episodes=5000)

if __name__ == "__main__":
    main()