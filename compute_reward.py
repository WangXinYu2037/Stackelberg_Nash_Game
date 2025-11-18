import numpy as np
import math
import cvxpy as cp
Loss = [[3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481],
    [4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391],
    [3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441],
     # [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001],
    [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601],
    [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]]
UAV_max_power = 4
Bandwidth = 0.5e6
# c_o = 0.9 * 0.5e6
c_o = 0.18 * 0.5e6
c_f = 0.12 * 0.5e6


def I_calculate(PJ1, PJ2, PN1, PN2, PN3, time_slot):
    PN = [PN1, PN2, PN3]
    PJ = [PJ1, PJ2]
    # print(PN1, PN2, PN3)
    # print(PN)
    # print(PJ1, PJ2)
    # print(PJ)
    I = []
    for t in range(time_slot):
        for i in range(3):
            tmp = 0
            for n in range(3):
                if n != i:
                    tmp += PN[n][t] / Loss[n][i]  # 来自其它无人机n对用户i的干扰
            for j in range(2):
                tmp += PJ[j][t] / Loss[3+j][i]  # 来自jammers的干扰
            I.append(tmp)  # 前三个是t=1时的干扰，后三个是t=2时的干扰
    return I


def compute_reward(action, time_slot):
    PJ1 = action[0:time_slot]
    PJ2 = action[time_slot:2*time_slot]

    PN1 = UAV_max_power * np.random.rand(1, time_slot).flatten()
    PN2 = UAV_max_power * np.random.rand(1, time_slot).flatten()
    PN3 = UAV_max_power * np.random.rand(1, time_slot).flatten()
    for _ in range(50):

        try:
            I = I_calculate(PJ1, PJ2, PN1, PN2, PN3, time_slot)  # t = 1和2 时的干扰
            # print("I:", I)
            PN1_var = cp.Variable(time_slot)
            uNt1 = Bandwidth * cp.log(1 + PN1_var[0] / (Loss[0][0] * I[0])) - c_f * PN1_var[0]
            uNt2 = Bandwidth * cp.log(1 + PN1_var[1] / (Loss[0][0] * I[3])) - c_f * PN1_var[1]
            prob = cp.Problem(cp.Maximize(uNt1 + 0.9 * uNt2), [[1, 1] @ PN1_var <= 7, PN1_var <= [UAV_max_power, UAV_max_power], PN1_var >= 0])
            prob.solve(verbose=False)
            PN1 = PN1_var.value

            # uN1 = Bandwidth * math.log(1 + PN1[0] / (Loss[0][0] * I[0])) - c_f * PN1[0] + 0.9 * Bandwidth * math.log(
            #     1 + PN1[1] / (Loss[0][0] * I[3])) - c_f * PN1[1]
            # print("第一次:", PN1.value)

            I = I_calculate(PJ1, PJ2, PN1, PN2, PN3, time_slot)
            PN2_var = cp.Variable(time_slot)
            uNt1 = Bandwidth * cp.log(1 + PN2_var[0] / (Loss[1][1] * I[1])) - c_f * PN2_var[0]
            uNt2 = Bandwidth * cp.log(1 + PN2_var[1] / (Loss[1][1] * I[4])) - c_f * PN2_var[1]
            prob = cp.Problem(cp.Maximize(uNt1 + 0.9 * uNt2), [[1, 1] @ PN2_var <= 7, PN2_var <= [UAV_max_power, UAV_max_power], PN2_var >= 0])
            prob.solve()
            PN2 = PN2_var.value
            # uN2 = Bandwidth * math.log(1 + PN2[0] / (Loss[0][0] * I[0])) - c_f * PN2[0] + 0.9 * Bandwidth * math.log(
            #     1 + PN2[1] / (Loss[0][0] * I[3])) - c_f * PN2[1]

            I = I_calculate(PJ1, PJ2, PN1, PN2, PN3, time_slot)
            PN3_var = cp.Variable(time_slot)
            uNt1 = Bandwidth * cp.log(1 + PN3_var[0] / (Loss[2][2] * I[2])) - c_f * PN3_var[0]
            uNt2 = Bandwidth * cp.log(1 + PN3_var[1] / (Loss[2][2] * I[5])) - c_f * PN3_var[1]
            prob = cp.Problem(cp.Maximize(uNt1 + 0.9 * uNt2), [[1, 1] @ PN3_var <= 7, PN3_var <= [UAV_max_power, UAV_max_power], PN3_var >= 0])
            prob.solve()
            PN3 = PN3_var.value
        except:

            continue
        # uN3 = Bandwidth * math.log(1 + PN3[0] / (Loss[0][0] * I[0])) - c_f * PN3[0] + 0.9 * Bandwidth * math.log(
        #     1 + PN3[1] / (Loss[0][0] * I[3])) - c_f * PN3[1]

        # print(_, PN1, PN2, PN3)
        # print(uN1, uN2, uN3)



    reward = 0
    # if type(PN1) == cp.Variable:
    #     PN1 = PN1.value
    # if type(PN2) == cp.Variable:
    #     PN2 = PN2.value
    # if type(PN3) == cp.Variable:
    #     PN3 = PN3.value
    for t in range(time_slot):
        reward -= pow(0.9, t) * Bandwidth * math.log(1 + PN1[t] / (Loss[0][0] * I[3 * t]))
        reward -= pow(0.9, t) * Bandwidth * math.log(1 + PN2[t] / (Loss[1][1] * I[3 * t + 1]))
        reward -= pow(0.9, t) * Bandwidth * math.log(1 + PN3[t] / (Loss[2][2] * I[3 * t + 2]))
        reward -= pow(0.9, t) * c_o * PJ1[t]
        reward -= pow(0.9, t) * c_o * PJ2[t]

    return reward