from random import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math
import gym

import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from compute_reward import compute_reward

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    evaluate_PN_reward = [0, 0, 0]
    for _ in range(times):
        # s = env.reset()
        s = env.state
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating

            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, PN_reward = env.step(action)
            print(action, s_, r, done)
            print("评估时选取的动作", a, "本次动作的奖励", r)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

        for i in range(len(evaluate_PN_reward)):
            evaluate_PN_reward[i] += PN_reward[i]

        evaluate_PN_reward = [evaluate_PN_reward[i]/times for i in range(len(evaluate_PN_reward))]
    return evaluate_reward / times, evaluate_PN_reward


def main(args, env_name, number, seed):
    # env = gym.make(env_name)
    env = Environment(args)
    env_evaluate = Environment(args)  # When evaluating the policy, we need to rebuild an environment
    print("评估环境的状态", env_evaluate.state)
    # Set random seed
    env.seed(seed)
    # env.action_space.seed(seed)
    env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    args.max_action = env.max_action
    args.jammers_num = env.jammers_num
    args.budget = env.budget
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_PN_rewards = []  # Record the rewards of PN1, PN2 and PN3
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        # s = env.reset()
        s = env.state
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, evaluate_PN_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_PN_rewards.append(evaluate_PN_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], evaluate_num)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
                    np.save('./data_train/PN_rewards_PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_PN_rewards))


class Environment:
    def __init__(self, args):
        # self.number_ED = 5
        # self.state_dim = 1
        # self.sigma = 2e-14  # 环境噪声
        self.size_model = 5e6  # 模型大小，单位bit
        self.size_data = 6e7  # 数据集大小，单位bit
        # self.ground_ED = 300  # 地面设备距中心的平均距离
        # self.num_ED = 3  # 地面设备的数量
        # self.UAV = 600  # UAV距中心的平均距离
        # self.num_UAV = 2  # UAV设备的数量
        # self.Bandwidth = 5e6  # 带宽，单位赫兹
        # self.phi_CPU = 80  # CPU处理每个比特所需的轮数
        # self.Rician = 15  # LoS的信道莱斯因子
        # self.g0 = 0.8  # LoS的单位信道增益
        # self.P_max = 0.2  # 最大发射功率
        # self.E_max = 0.04  # 最大能量消耗
        # self.f_max = 25e8  # 处理器的最大频率
        self._max_episode_steps = 30
        self.policy_dist = args.policy_dist

        self.state = [3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481,
    4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391,
    3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441,
     # [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001],
    3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601,
    3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]
        # self.state = [2, 8, 18, 18]
        self.action = None
        self.jammers_num = 2  # jammers的数量
        self.UAV_num = 3  # UAV BSs的数量

        self.state_dim = len(self.state)
        self.time_slot = 2
        self.action_dim = self.time_slot * self.jammers_num  # 时间维度(timeslot) * 数量
        self.max_action = 2  # jammers的最大功率
        self.budget = [3, 3.5]  # jammers的能量预算

        self.distance = None
        self.kappa = 1e-27


    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        # print(action)
        reward, PN_reward = compute_reward(action, self.time_slot)

        # x, y = action[0], action[1]
        # a, b, c, d = self.state[0], self.state[1], self.state[2], self.state[3]
        # reward = -a * pow(x, 2) + b * x - c * pow(y, 2) + d * y
        return self.state, reward, True, PN_reward

    # def step(self, action):
    #     """在环境中执行动作返回新的状态、奖励和结束标识符"""
    #     # 从动作中分离功率和频率
    #     power = action[0:5]
    #     frequency = action[5:10]
    #     # 计算信道增益
    #     g_i = [self.g0 / ((1 + self.Rician) * i ** 2) for i in self.distance]
    #     R = []
    #     # 总设备的数量
    #     total = self.num_UAV + self.num_ED
    #     # 逐个计算NOMA的上行传输速率，存储在列表中
    #     temp = self.sigma
    #     for i in range(total):
    #         for j in range(i+1, total):
    #             temp += power[j] * g_i[j] ** 2
    #         r = self.Bandwidth * math.log2(1 + power[i] * g_i[i] ** 2 / temp)
    #         R.append(r)
    #     # 计算上行时间
    #     T_up = [self.size_model / r_i for r_i in R]
    #     # 计算训练时间
    #     T_tra = [self.size_data * self.phi_CPU / f_i for f_i in frequency]
    #     # 计算能耗
    #     E_i = [self.kappa * self.phi_CPU * self.size_data * f_i ** 2 for f_i in frequency]
    #     # 计算能耗和功率的惩罚项，保证其在最大能耗、功率的范围内
    #     E_punish = sum(E_i) - self.E_max * total
    #     # 计算reward
    #     reward = -(max(T_up) + max(T_tra) + E_punish)
    #     # 计算一个距离的扰动，用正态分布计算，距离扰动范围限制在[-5, 5]
    #     disturbance = []
    #     # while len(disturbance) < len(self.distance):
    #     #     dis = round(np.random.normal(0, 5))
    #     #     if -5 <= dis <= 5:
    #     #         disturbance.append(dis)
    #     # self.distance = [x + y for x, y in zip(self.distance, disturbance)]
    #     action = np.concatenate((action, self.distance))
    #     return action, reward, True

    def reset(self):
        # 距离采样
        # sample_distance = []
        # sample_power = []
        # sample_frequency = []
        #
        # while len(sample_power) < (self.num_ED + self.num_UAV):
        #     power = np.random.normal()
        #     if -1 <= power <= 1 and power != 0:
        #         sample_power.append(np.abs(power * self.P_max))
        # while len(sample_frequency) < (self.num_ED + self.num_UAV):
        #     fre = np.random.normal()
        #     if -1 <= fre <= 1 and fre != 0:
        #         sample_frequency.append(np.abs(fre * self.f_max))
        # while len(sample_distance) < self.num_ED:
        #     distance = np.random.normal(self.ground_ED, 10)
        #     if np.abs(distance-self.ground_ED) <= 0.5:
        #         sample_distance.append(np.abs(distance))
        # while len(sample_distance) < (self.num_ED + self.num_UAV):
        #     distance = np.random.normal(self.UAV, 10)
        #     if np.abs(distance-self.UAV) <= 0.5:
        #         sample_distance.append(np.abs(distance))
        # self.distance = sample_distance
        #
        # self.state = sample_power + sample_frequency + self.distance
        # self.action = self.state
        self.state = np.random.randint(1, 21, size=len(self.state))

        return self.state


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(5e3), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_index = 1
    main(args, env_name=env_name[env_index], number=1, seed=10)
