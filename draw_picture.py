import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d
import pandas as pd

def UAV_compare():
# 修改为你的实际 npy 文件路径
    file_path = './data_train/PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
    file_path2 = './data_train/PN_rewards_PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'

    file_path_opt = './data_train/数据备份/对照组数据_有优化/PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
    file_path2_opt = './data_train/数据备份/对照组数据_有优化/PN_rewards_PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'

    # 读取 npy 文件
    evaluate_rewards = np.load(file_path)[::10]
    evaluate_PN_rewards = np.load(file_path2)
    pn1 = evaluate_PN_rewards[:, 0]
    pn2 = evaluate_PN_rewards[:, 1]
    pn3 = evaluate_PN_rewards[:, 2]

    evaluate_rewards_opt = np.load(file_path_opt)
    evaluate_PN_rewards_opt = np.load(file_path2_opt)
    pn1_opt = evaluate_PN_rewards_opt[:, 0]
    pn2_opt = evaluate_PN_rewards_opt[:, 1]
    pn3_opt = evaluate_PN_rewards_opt[:, 2]

    x = np.arange(evaluate_rewards.shape[0])

    # 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Jammers效益绘图----------------------------------

    plt.figure()
    plt.plot(x, evaluate_rewards_opt, 'r', label='Jammers_SSNG', linewidth=2)
    plt.plot(x, evaluate_rewards, 'b', label='Jammers_PPO', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Rewards', fontsize=12)
    # plt.ylim([-1.4e6, -0.4e6])
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # 强制始终使用科学计数法
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # UAV效益对比 绘图 --------------------------------
    plt.figure()
    pn1_opt = uniform_filter1d(pn1_opt, size=20)
    pn2_opt = uniform_filter1d(pn2_opt, size=20)
    pn3_opt = uniform_filter1d(pn3_opt, size=30)
    plt.plot(pn2_opt, 'b', linewidth=1.5, label=r'$N_1$-SSNG')
    plt.plot(pn3_opt, 'g', linewidth=1.5, label=r'$N_2$-SSNG')
    plt.plot(pn1_opt, 'r', linewidth=1.5,  label=r'$N_3$-SSNG')

    pn1 = pn1[::10]
    pn2 = pn2[::10]
    pn3 = pn3[::10]
    pn1[0] = pn1_opt[0]
    pn2[0] = pn2_opt[0]
    pn3[0] = pn3_opt[0]
    # pn1 = uniform_filter1d(pn1, size=5)
    # pn2 = uniform_filter1d(pn2, size=5)
    # pn3 = uniform_filter1d(pn3, size=5)
    plt.plot(pn2, 'b', linewidth=1.5, label=r'$N_1$-Static', linestyle='-.')
    plt.plot(pn3, 'g', linewidth=1.5, label=r'$N_2$-Static', linestyle='-.')
    plt.plot(pn1, 'r', linewidth=1.5,  label=r'$N_3$-Static', linestyle='-.')




    plt.xlabel('Evaluate Episode in SSNG-PPO', fontsize=12)
    plt.ylabel('Cumulative Rewards of Jammers', fontsize=12)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # 强制始终使用科学计数法
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.grid(True)

    plt.legend(fontsize=11, loc='best')

    plt.tight_layout()
    # plt.show()
    plt.savefig('UAV comparison.pdf', bbox_inches='tight', dpi=300)



def Jammer_compare():

    file_path_opt = './data_train/数据备份/学习率200/lr5e4PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
    file_path_opt_1 = './data_train/数据备份/学习率200/lr1e4PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
    file_path_opt_2 = './data_train/数据备份/学习率200/lr5e5PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
    evaluate_rewards_opt = np.load(file_path_opt)
    evaluate_rewards_opt_1 = np.load(file_path_opt_1)
    evaluate_rewards_opt_2 = np.load(file_path_opt_2)
    x = np.arange(evaluate_rewards_opt.shape[0])

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式用 STIX（接近 Times）
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    # Jammers绘图

    plt.figure()
    # evaluate_rewards_opt = uniform_filter1d(evaluate_rewards_opt, size=10)
    # evaluate_rewards_opt_1 = uniform_filter1d(evaluate_rewards_opt_1, size=10)
    evaluate_rewards_opt_2 = uniform_filter1d(evaluate_rewards_opt_2, size=3)
    plt.plot(x, evaluate_rewards_opt, label=r'Learning rate$=5\times 10^{-4}$', linewidth=1.5)
    plt.plot(x, evaluate_rewards_opt_1, label=r'Learning rate$=1\times 10^{-4}$', linewidth=1.5)
    plt.plot(x, evaluate_rewards_opt_2, label=r'Learning rate$=2\times 10^{-5}$', linewidth=1.5)
    # plt.plot(x, evaluate_rewards_opt_1, label=r'Learning rate $=1\times 10^{-4}$', linewidth=1)

    plt.xlabel('Evaluate Episode in SSNG-PPO', fontsize=12)
    plt.ylabel('Cumulative Rewards of UAV-BSs', fontsize=12)
    # plt.ylim([-1.4e6, -0.4e6])
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # 强制始终使用科学计数法
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Jammers comparison.pdf', bbox_inches='tight', dpi=300)


Jammer_compare()
# UAV_compare()


# Step 1: 加载 npy 文件
# npy_path = './data_train/数据备份/学习率200/lr1e4PPO_continuous_Gaussian_env_HalfCheetah-v2_number_1_seed_10.npy'
# data = np.load(npy_path)
#
# # Step 2: 转成 DataFrame
# df = pd.DataFrame(data)
#
# # Step 3: 保存为 Excel
# excel_path = './converted.xlsx'
# df.to_excel(excel_path, index=False)
#
# print("转换完成！Excel 保存为：", excel_path)



# Step 1: 读取你修改后的 Excel
# excel_modified_path = './converted_modified.xlsx'
# df_modified = pd.read_excel(excel_modified_path)
#
# # Step 2: 转为 numpy 数组
# data_modified = df_modified.to_numpy()
#
# # Step 3: 保存为 npy
# npy_output_path = './modified.npy'
# np.save(npy_output_path, data_modified)
#
# print("回转完成！新 npy 保存为：", npy_output_path)