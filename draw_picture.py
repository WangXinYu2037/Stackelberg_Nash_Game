import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 修改为你的实际 npy 文件路径
file_path = './data_train/PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
file_path2 = './data_train/PN_rewards_PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'

# file_path = './data_train/数据备份/对照组数据_有优化/PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
# file_path2 = './data_train/数据备份/对照组数据_有优化/PN_rewards_PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'

# 读取 npy 文件
evaluate_rewards = np.load(file_path)
evaluate_PN_rewards = np.load(file_path2)
pn1 = evaluate_PN_rewards[:, 0]
pn2 = evaluate_PN_rewards[:, 1]
pn3 = evaluate_PN_rewards[:, 2]

x = np.arange(evaluate_rewards.shape[0])

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Jammers绘图
# plt.figure()
# plt.plot(x, evaluate_rewards, label=r'learning rate = $1.5\times10^{-4}$', linewidth=2)
# plt.xlabel('Episode', fontsize=12)
# plt.ylabel('Accumulative Rewards', fontsize=12)
# # plt.title('Evaluate Rewards Curve', fontsize=12)
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

# UAV 绘图
plt.figure()
plt.plot(pn1, linewidth=2, label='N1')
plt.plot(pn2, linewidth=2, label='N2')
plt.plot(pn3, linewidth=2, label='N2')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Accumulative Rewards', fontsize=12)

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # 强制始终使用科学计数法
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()
