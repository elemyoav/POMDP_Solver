from envs.tiger import Tiger
from envs.box_pushing import BoxPushing

from architectures.DTQN import Agent as DTQNAgent
from architectures.DRQN import Agent as DRQNAgent

from args import args

def main():
    env = choose_env(args.env)
    agent = choose_agent(args.model, env)
    agent.train(max_episodes=args.max_episodes)

def choose_env(env_name):
    if env_name == 'Tiger':
        return Tiger()
    elif env_name == 'BoxPushing':
        return BoxPushing()
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