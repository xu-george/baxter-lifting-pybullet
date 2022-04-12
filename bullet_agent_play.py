import argparse
import os
import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

from baxter_bullet_env.baxter_gym import BaxterGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # --------------------environment ---------------------
    parser.add_argument('--env-name', default='Baxter_bullet_Lift')
    parser.add_argument('--epoch_step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 10000)

    # using dense reward, and physic state to train
    env = BaxterGymEnv(renders=True, camera_view=False,
                       pygame_renders=False, max_episode_steps=args.epoch_step)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = torch.load("./checkpoints/eposiod9900_reward:134.75771245244255_model.pt")

    episodes = 20

    for episode in range(episodes):
        print("testing episode_{}".format(episode))
        state = env.reset()
        step = 0
        episode_reward = 0
        done = False
        while not done and step < args.epoch_step:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward
            step += 1
            state = next_state
    env.close()

