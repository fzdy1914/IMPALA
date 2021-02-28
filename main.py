import torch
import argparse
import torch.multiprocessing as mp

from model import DenseNet
from learner import learner
from actor import actor
from environment import EnvironmentProxy
from utils import ParameterServer

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=6,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=123,
                        help="the seed of random, default is 123")
    parser.add_argument('--length', type=int, default=16,
                        help='Number of steps to run the agent')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of steps to run the agent')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--entropy_cost", type=float, default=0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Learning rate, default is 0.00048")
    parser.add_argument("--decay", type=float, default=0.00001,
                        help="RMSProp optimizer decay, default is .99")
    parser.add_argument('--save_path', type=str, default="./state/epoch-%s.pt",
                        help='Set the path to save trained model parameters')
    parser.add_argument('--load_path', type=str, default="./state/epoch-0.pt",
                        help='Set the path to load trained model parameters')

    args = parser.parse_args()
    data = mp.Queue(maxsize=32)
    lock = mp.Lock()
    env_args = {'debug': False}
    action_size = 4
    args.action_size = action_size

    is_training_done = mp.Event()
    is_training_done.clear()
    qs = [mp.Queue(maxsize=10) for _ in range(args.actors)]

    model = DenseNet()
    envs = [EnvironmentProxy(env_args) for idx in range(args.actors)]

    learner = mp.Process(target=learner, args=(model, data, qs, is_training_done, args))

    actors = [mp.Process(target=actor, args=(idx, qs[idx], data, envs[idx], is_training_done, args))
              for idx in range(args.actors)]
    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()
