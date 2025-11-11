import math
import random
import numpy as np
import matplotlib.pyplot as plt

def I_calculate(PN, PJ):
    # calculate interference Ii
    I = [0 for _ in range(MD)]
    for i in range(MD):
        tmp = 0
        for n in range(N):
            if n != i:
                tmp += PN[n] / Loss[n][i]
        for j in range(J):
            tmp += PJ[j] / Loss[4 + j][i]
        I[i] = tmp
    return I


def uN_calculate(PN, I):
    RN = [0 for _ in range(N)]
    uN = [0 for _ in range(N)]
    for i in range(N):
        # print(Loss[i][i], sigma2, I[i])
        RN[i] = Bandwidth * math.log(1 + PN[i] / (Loss[i][i] * (sigma2 + I[i])), 2)
        # print(RN[i])
        uN[i] = round(RN[i] - E * PN[i], 3)
        # print(uN[i])
    return RN, uN


def Best_response(PN, PJ, rounds):
    for t in range(rounds):
        print(PN)
        I = I_calculate(PN, PJ)
        RN, uN = uN_calculate(PN, I)
        # print("干扰", I)
        # print("速率", RN)
        # print("效用", uN)

        # 迭代顺序不影响结果
        # for i in range(N - 1, -1, -1):
        #     # print(Loss[i][i] *(sigma2 + I[i]))
        #     PN[i] = min(max(Bandwidth / E - Loss[i][i] * (sigma2 + I[i]), 0), P_max)
        #     I = I_calculate(PN, PJ)
        for i in range(N):
            # print("最值", Loss[i][i] * (sigma2 + I[i]))
            PN[i] = min(max(Bandwidth / E - Loss[i][i] * (sigma2 + I[i]), 0), P_max)
            I = I_calculate(PN, PJ)
        return PN

N, J = 3, 2  # UAV BSs and jammers
MD = N
P_max = 1  # max transmission power
Bandwidth = 0.5e6  # Hz
n0 = -170  # dBm/Hz 计算得到的白噪声功率近似为0
sigma2 = pow(10, n0 * Bandwidth / 10) * 0.001  # W
a, b = 11.95, 0.14
uNLoS, uLoS = 23, 3  # dB
fc = 2000  # carrier frequency Hz
c = 3e8  # light speed m/s
H = 150  # Height m

uNloS, uLoS = pow(10, uNLoS / 10), pow(10, uLoS / 10)
Ko = (4 * math.pi * fc / c) ** 2

Nxy = [(25, 25), (25, 75), (75, 25), (75, 75)]
Jxy = [(25, 50), (75, 50)]
MDxy = [(36, 18), (24, 52), (54, 33), (81, 94)]

Loss = [[0 for _ in range(N)] for _ in range(N + J)]

# compute the loss between UAV BSs and MDs
# for i in range(N):
#     for j in range(MD):
#         Ni, MDj = Nxy[i], MDxy[j]
#         dij = (Ni[0] - MDj[0], Ni[1] - MDj[1])  # 2D vector
#         Dij2 = H ** 2 + dij[0] ** 2 + dij[1] ** 2  # distance square between UAV BS i and MD j
#         theta_ij = math.asin(H / math.sqrt(Dij2))  # angle
#         PLoS = 1 / (1 + a * math.exp(-b * (theta_ij - a)))
#         PNLoS = 1 - PLoS
#         Loss[i][j] = Ko * Dij2 * (uNLoS * PNLoS + uLoS * PLoS)
#
# # compute the loss between Jammers and MDs
# for i in range(J):
#     for j in range(MD):
#         Ji, MDj = Jxy[i], MDxy[j]
#         dij = (Ji[0] - MDj[0], Ji[1] - MDj[1])  # 2D vector
#         Dij2 = H ** 2 + dij[0] ** 2 + dij[1] ** 2  # distance square between UAV BS i and MD j
#         theta_ij = math.asin(H / math.sqrt(Dij2))  # angle
#         PLoS = 1 / (1 + a * math.exp(-b * (theta_ij - a)))
#         PNLoS = 1 - PLoS
#         Loss[N + i][j] = Ko * Dij2 * (uNLoS * PNLoS + uLoS * PLoS)
#
# print(Loss)
Loss = [[3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481],
        [4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391],
        [3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441],
        [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001],
        [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601],
        [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]]


# for i in range(N + J):
#     for j in range(N):
#         Loss[i][j] = 0.001 * random.random()
#
# print(Loss)

# PN = np.random.uniform(0, P_max, N)
# PJ = np.random.uniform(0, P_max, J)

# PN = [0.13217978, 0.75608464, 0.63966809, 0.31077573]
PN = [0.13217978, 0.35608464, 0.63966809]
# PN = [0.83217978, 0.55608464, 0.63966809]
PJ = [0.899520962, 0.86197976]


E = 0.3 * 0.5e6  # energy cost


I = I_calculate(PN, PJ)  # calculate interference
RN, uN = uN_calculate(PN, I)  # calculate utility
print("干扰", I)
print("功率", PN)
print("接收功率")
for i in range(N):
    print(PN[i] / Loss[i][i], end=' ')
print()
print("速率", RN)
print("效用", uN)

print("最佳响应迭代----------------------------------------")
y1, Y1 = [], []
y2, Y2 = [], []
y3, Y3 = [], []
T = 180
for t in range(T):
    print(PN)
    y1.append(PN[0])
    y2.append(PN[1])
    y3.append(PN[2])

    I = I_calculate(PN, PJ)
    RN, uN = uN_calculate(PN, I)

    Y1.append(uN[0])
    Y2.append(uN[1])
    Y3.append(uN[2])

    print("干扰", I)
    # print("速率", RN)
    # print("效用", uN)

    #迭代顺序不影响结果
    # for i in range(N - 1, -1, -1):
    #     # print(Loss[i][i] *(sigma2 + I[i]))
    #     PN[i] = min(max(Bandwidth / E - Loss[i][i] * (sigma2 + I[i]), 0), P_max)
    #     I = I_calculate(PN, PJ)
    for i in range(N):
        print("最值", Loss[i][i] * (sigma2 + I[i]))
        PN[i] = min(max(Bandwidth / E - Loss[i][i] * (sigma2 + I[i]), 0), P_max)
        I = I_calculate(PN, PJ)



print("验证纳什均衡特性---------------------(失败，效果不明显")
# 对于图一，偏离N1的动作

fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
x = np.linspace(0, T, T)
ax[0].plot(x, y1)
ax[0].plot(x, y2)
ax[0].plot(x, y3)
fig.show()

