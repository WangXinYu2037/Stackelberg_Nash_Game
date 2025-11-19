import numpy as np
import matplotlib.pyplot as plt

# 修改为你的实际 npy 文件路径
file_path = './data_train/PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'

# 读取 npy 文件
evaluate_rewards = np.load(file_path)
x = np.arange(evaluate_rewards.shape[0])

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘图
plt.figure()
plt.plot(x, evaluate_rewards, label=r'learning rate = $1.5\times10^{-4}$', linewidth=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Accumulative Rewards', fontsize=12)
# plt.title('Evaluate Rewards Curve', fontsize=14)
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()
