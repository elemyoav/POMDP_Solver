import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DRQN')
parser.add_argument('--env', type=str, default='BoxPushing')
parser.add_argument('--max_episodes', type=int, default=1500)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--time_steps', type=int, default=4)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9995)
parser.add_argument('--eps_min', type=float, default=0.01)
args = parser.parse_args()