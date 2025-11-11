function [PN] = Best_response(PN, PJ, rounds)
P_max = 1;  % max transmission power
Bandwidth = 0.5e6;  % Hz
n0 = -170;  % dBm/Hz 计算得到的白噪声功率近似为0
sigma2 = 10 .^ (n0 * Bandwidth / 10) * 0.001;  % W
a = 11.95;
b = 0.14;
uNLoS = 23;
uLoS = 3;  % dB
fc = 2000;  % carrier frequency Hz
c = 3e8;  % light speed m/s
H = 150;  % Height m

for t = 1: rounds
    I = I_calcu(PN, PJ);
    for i = 1: N
        PN(i) =  min(max(Bandwidth / E - Loss(i, i) * (sigma2 + I(i)), 0), P_max);
        I = I_calcu(PN, PJ);
    end
end
end