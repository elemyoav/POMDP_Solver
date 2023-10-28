from envs.tiger import Tiger
from model.DTQN import Agent as DTQNAgent
from model.DRQN import Agent as DRQNAgent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DTQN')
parser.add_argument('--env', type=str, default='Tiger')
parser.add_argument('--max_episodes', type=int, default=1500)
args = parser.parse_args()

def main():
    env = choose_env(args.env)
    agent = choose_agent(args.model, env)
    agent.train(max_episodes=args.max_episodes)

def choose_env(env_name):
    if env_name == 'Tiger':
        return Tiger()
    else:
        raise NotImplementedError
    
def choose_agent(model_name, env):
    if model_name == 'DTQN':
        return DTQNAgent(env)
    elif model_name == 'DRQN':
        return DRQNAgent(env)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()