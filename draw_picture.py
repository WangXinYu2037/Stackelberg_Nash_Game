import numpy as np
import matplotlib.pyplot as plt

# 修改为你的实际 npy 文件路径
file_path = './data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(
    'normal', 'env1', 1, 0
)

# 读取 npy 文件
evaluate_rewards = np.load(file_path)

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘图
plt.figure()
plt.plot(evaluate_rewards)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Evaluate Rewards', fontsize=12)
plt.title('Evaluate Rewards Curve', fontsize=14)
plt.grid(True)

plt.show()
