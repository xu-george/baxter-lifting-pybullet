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
    parser.add_argument('--env-name', default='BaxterPaddleGrasp_auto_gamma_0.99')
    parser.add_argument('--epoch_step', default=100)

    # -------------------sac agent -------------------------
    parser.add_argument('--end_epoch', type=int, default=1e4)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.985)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.03)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.3, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.3)')
    parser.add_argument('--alpha_lr', type=float, default=1e-4, metavar='G',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1e8, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 1024)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                        help='Steps sampling random actions (default: 2000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1e6, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 10000)

    # ------------------------load controllers

    # using dense reward, and physic state to train
    env = BaxterGymEnv(renders=True, camera_view=False,
                       pygame_renders=False, max_episode_steps=args.epoch_step)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # load checkpoint
    # agent.load_checkpoint(os.getcwd() + "/checkpoints/sac_checkpoint_BaxterPaddleGrasp_auto_gamma_0.99_")

    # Tesnorboard
    writer = SummaryWriter('log/SAC_{}_{}_{}_{}'.format(args.env_name, args.seed,
                                                        args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_avg_reward = -500
    for i_episode in range(1, int(args.end_epoch+1)):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while episode_steps < args.epoch_step and not done:
            
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action)
            env.render()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)    # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format
              (i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # evaluate agent
        if i_episode % 20 == 0 and args.eval is True:
            print("Test the agent:{}".format(i_episode))
            avg_reward = 0.
            episodes = 5
            for _ in range(episodes):
                env.reset()
                test_step = 0
                episode_reward = 0
                done = False
                while test_step < args.epoch_step:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = env.step(action)

                    env.render()

                    episode_reward += reward
                    test_step += 1
                    state = next_state
                avg_reward += episode_reward

            avg_reward /= episodes

            # save the agent
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_checkpoint(args.env_name)
                torch.save(agent, "./checkpoints/eposiod{}_reward:{}_model.pt".format(i_episode, avg_reward))

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(i_episode, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()

