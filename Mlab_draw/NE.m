
close all;
Loss = [[3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481];
        [4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391];
        [3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441];
%         [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001];
        [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601];
        [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]];
    
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
E = 0.3 * 0.5e6;

PN = [0.76217978, 0.55608464, 0.33966809];
PJ = [0.899520962, 0.86197976];
% PJ = [1.15144 1.99996];
N = length(PN);
J = length(PJ);

T = 300;
y1 = zeros(1, T);
y2 = zeros(1, T);
y3 = zeros(1, T);
% y4 = zeros(1, T);

Y1 = zeros(1, T);
Y2 = zeros(1, T);
Y3 = zeros(1, T);
% Y4 = zeros(1, T);

% 展示收敛到NE的过程
for t = 1: T
    I = I_calcu(PN, PJ);
    [RN, uN] = uN_calcu(PN, I);
    
    y1(t) = PN(1);
    y2(t) = PN(2);
    y3(t) = PN(3);
%     y4(t) = PN(4);

    for i = 1: N
        PN(i) =  min(max(Bandwidth / E - Loss(i, i) * (sigma2 + I(i)), 0), P_max)
        I = I_calcu(PN, PJ);
    end
end

PN = [0.53217978, 0.35608464, 0.26966809];
for t = 1: T
    I = I_calcu(PN, PJ);
    [RN, uN] = uN_calcu(PN, I);
    
    Y1(t) = PN(1);
    Y2(t) = PN(2);
    Y3(t) = PN(3);
%     Y4(t) = PN(4);
    for i = 1: N
        PN(i) =  min(max(Bandwidth / E - Loss(i, i) * (sigma2 + I(i)), 0), P_max);
        I = I_calcu(PN, PJ);
    end
end

x = linspace(1, T, T);
figure(1);

subplot(1, 2, 1);
plot(x, y1, 'LineStyle','-', 'LineWidth',2, 'Displayname','N1');
hold on;
grid on;
plot(x, y2, 'LineStyle','-.', 'LineWidth',2, 'Displayname','N2');
plot(x, y3, 'LineStyle',':', 'LineWidth',2, 'Displayname', 'N3');
% plot(x, y3, 'Displayname', 'N4');
ylim([0.2 1]);
xlabel('Iteration'); ylabel('Power (W)');
legend;
hold off;
% figure(2);
subplot(1, 2, 2);
plot(x, Y1, 'LineStyle','-', 'LineWidth',2, 'Displayname','N1');
hold on;
grid on;
plot(x, Y2, 'LineStyle','-.', 'LineWidth',2, 'Displayname','N2');
plot(x, Y3, 'LineStyle',':', 'LineWidth',2,'Displayname', 'N3');
% plot(x, Y4, 'Displayname', 'N4');
ylim([0.2 1]);
xlabel('Iteration'); ylabel('Power (W)');
legend;
hold off;
