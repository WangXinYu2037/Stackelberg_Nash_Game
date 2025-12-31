%%
clc;clear;close
P_max = 1;  % max transmission power
Bandwidth = 0.5e6;  % Hz
n0 = -170;  % dBm/Hz 计算得到的白噪声功率近似为0
sigma2 = 10 .^ (n0 * Bandwidth / 10) * 0.001;  % W   错了，这里计算过程大错特错了，应该是-170 + 10log(0.5e6, 10) dBm，不过反正都很小，不影响
a = 11.95;
b = 0.14;
uNLoS = 23;
uLoS = 3;  % dB
fc = 2000;  % carrier frequency Hz
c = 3e8;  % light speed m/s
H = 150;  % Height m
E = 0.3 * 0.5e6;
T = 300;

Loss = [[3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481];
        [4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391];
        [3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441];
%         [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001];
        [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601];
        [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]];
 
% PN = [   0.53217978, 0.95608464, 0.26966809
%           0.53217978, 0.95608464, 0.26966809];
PN = [       0.5643    0.3378    0.8137
    0.4315    0.7207    0.3741];
% PN = [       0.5643    0.7378    0.4137
%     0.7315    0.4507    0.8741];
PJ = [0.899520962, 0.66197976];

P1_T = zeros(T+1, 2);
P2_T = zeros(T+1, 2);
P3_T = zeros(T+1, 2);
u1_T = zeros(T, 1);
u2_T = zeros(T, 1);
u3_T = zeros(T, 1);

P1_T(1, :) = PN(:, 1);
P2_T(1, :) = PN(:, 2);
P3_T(1, :) = PN(:, 3);
for t= 1: T
    I1 = I_calcu(PN(1, :), PJ); % t=1的干扰
    I2 = I_calcu(PN(2, :), PJ); % t=2的干扰
    I = [ I1
          I2 ];
      
    cvx_begin
        variable P1(2);
        uN1 = Bandwidth * log(1 + P1(1) / (Loss(1,1) * I1(1)) ) - E * P1(1);
        uN2 = Bandwidth * log(1 + P1(2) / (Loss(1,1) * I2(1)) ) - E * P1(2);

        minimize( -(uN1 + 0.9 * uN2));
        subject to
            [0; 0] <= P1 <= [1; 1];
            [1, 1] * P1 <= 1.5;
    cvx_end;
    P1
    P1_T(t+1, :) = P1;
    PN(:, 1) = P1;
    u1_T(t) = uN1 + 0.9 * uN2;
    % ------------------------------------------------
    I1 = I_calcu(PN(1, :), PJ); % t=1的干扰
    I2 = I_calcu(PN(2, :), PJ); % t=2的干扰
    I = [ I1
          I2 ];
      
    cvx_begin
        variable P2(2); % 长度为2的向量
        uN1 = Bandwidth * log(1 + P2(1) / (Loss(2,2) * I1(2)) ) - E * P2(1);
        uN2 = Bandwidth * log(1 + P2(2) / (Loss(2,2) * I2(2)) ) - E * P2(2);

        minimize( -(uN1 + 0.9 * uN2));
        subject to
            [0; 0] <= P2 <= [1; 1];
            [1, 1] * P2 <= 1.4;
    cvx_end;
    P2
    P2_T(t+1, :) = P2;
    PN(:, 2) = P2;
    u2_T(t) = uN1 + 0.9 * uN2;
    % ------------------------------------------------
    I1 = I_calcu(PN(1, :), PJ); % t=1的干扰
    I2 = I_calcu(PN(2, :), PJ); % t=2的干扰
    I = [ I1
          I2 ];
    cvx_begin
        variable P3(2);
        uN1 = Bandwidth * log(1 + P3(1) / (Loss(3,3) * I1(3)) ) - E * P3(1);
        uN2 = Bandwidth * log(1 + P3(2) / (Loss(3,3) * I2(3)) ) - E * P3(2);

        minimize( -(uN1 + 0.9 * uN2));
        subject to
            [0; 0] <= P3 <= [1; 1];
            [1, 1] * P3 <= 1.8;
    cvx_end;
    P3
    P3_T(t+1, :) = P3;
    PN(:, 3) = P3;
    u3_T(t) = uN1 + 0.9 * uN2;
end
%% 绘制图4
figure(1);
hold on; grid on;
plot(P1_T(:, 1), P1_T(:, 2), 'LineWidth',2, 'Displayname','N1');
text(P1_T(T+1, 1), P1_T(T+1, 2), {'★'}, 'FontSize',12, 'HorizontalAlignment','center', 'color', '#8B0000');
text(P1_T(T+1, 1)+0.05, P1_T(T+1, 2), '$\mathbf{(0.267,0.284)}$', 'FontSize',12, 'FontName','Times New Roman', 'FontWeight','bold');

plot(P2_T(:, 1), P2_T(:, 2), 'LineWidth',2, 'Displayname','N2');
text(P2_T(T+1, 1), P2_T(T+1, 2), {'★'}, 'FontSize',12, 'HorizontalAlignment','center', 'color', '#8B0000');
text(P2_T(T+1, 1)-0.06, P2_T(T+1, 2) - 0.08, '$\mathbf{(0.716,0.683)}$', 'FontSize',12, 'FontName','Times New Roman', 'FontWeight','bold');

plot(P3_T(:, 1), P3_T(:, 2), 'LineWidth',2, 'Displayname','N3');
text(P3_T(T+1, 1), P3_T(T+1, 2), {'★'}, 'FontSize',12, 'HorizontalAlignment','center', 'color', '#8B0000');
text(P3_T(T+1, 1) - 0.2, P3_T(T+1, 2), '$\mathbf{(0.883,0.896)}$', 'FontSize',12, 'FontName','Times New Roman', 'FontWeight','bold');
xlabel('Transmit Power $(W)$ when $t = 1$', 'Interpreter', 'latex', 'FontName','Times New Roman'); 
ylabel('Transmit Power $(W$) when $t = 2$', 'Interpreter', 'latex', 'FontName','Times New Roman');
xlim([0.25 1]); ylim([0.25 1]);
legend('$N_1$','$N_2$', '$N_3$','Interpreter','latex', 'location', 'best');
% 
ax1 = axes('Position', [0.2 0.66 0.25 0.25]);
hold on;
plot(P2_T(:, 1), P2_T(:, 2), 'LineWidth',2, 'Displayname','N2', 'color', '#D95319');
text(P2_T(T+1, 1), P2_T(T+1, 2), {'★'}, 'FontSize',12, 'HorizontalAlignment','center', 'color', '#8B0000');
plot(P3_T(:, 1), P3_T(:, 2), 'LineWidth',2, 'Displayname','N3',  'color', '#EDB120');

xlim([0.68 0.720]); ylim([0.68 0.72]);
grid on;
set(ax1, 'Box', 'on')
%% 绘制图5
Tx = [1:1:300];
plot(Tx, u1_T, 'LineWidth',2, 'LineStyle','-', 'Displayname','N1', 'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', 1:30:length(Tx));
hold on; grid on;
text(110,3.2e4, 'Game Iteration $=220$','FontSize',12, 'FontName','Times New Roman', 'FontWeight','bold');
plot(Tx, u2_T, 'LineWidth',2, 'LineStyle','-.','Displayname','N2', 'Marker', 'd',  'MarkerSize', 5, 'MarkerIndices', 1:30:length(Tx));
plot(Tx, u3_T, 'LineWidth',2, 'LineStyle',':', 'Displayname','N3', 'Marker', 's',  'MarkerSize', 5, 'MarkerIndices', 1:30:length(Tx));
legend('$N_1$','$N_2$', '$N_3$','Interpreter','latex', 'location', 'east','fontsize',12);
xlabel('Game Iteration'); ylabel('Utility for UAV-BSs');
set(gca,'FontName','Times New Roman','fontsize',12)
xline(220,'LineWidth',2,'Color','#8B0000','HandleVisibility','off');
figure(1);

% arrowPlot(P1_T(:,1), P1_T(:,2), 'number', 3, 'LineWidth', 1)
% hold on;
% grid on;
% arrowPlot(P2_T(:,1), P2_T(:,2), 'number', 3, 'LineWidth', 1)
% arrowPlot(P3_T(:,1), P3_T(:,2), 'number', 3, 'LineWidth', 1)
% cmap = jet(T);
% hold on
% for t = 1:T
% %     plot(P1_T(t, 1), P1_T(t, 2), 'o', 'Color', cmap(t,:), 'MarkerFaceColor', cmap(t,:));
%     if t > 1
%         plot([P1_T(t-1, 1) P1_T(t, 1)], [P1_T(t-1, 2) P1_T(t, 2)], '-', 'Color', cmap(t,:),  'LineWidth',2);
%     end
% end
% arrow([P1_T(T-1, 1) P1_T(T-1, 2)], [P1_T(T, 1) P1_T(T, 2)])

% for t = 1:T
% %     plot(P2_T(t, 1), P2_T(t, 2), 'o', 'Color', cmap(t,:), 'MarkerFaceColor', cmap(t,:));
%     if t > 1
%         plot([P2_T(t-1, 1) P2_T(t, 1)], [P2_T(t-1, 2) P2_T(t, 2)], '-', 'Color', cmap(t,:),  'LineWidth',2);
%     end
% end
% arrow([P2_T(T-1, 1) P2_T(T-1, 2)], [P2_T(T, 1) P2_T(T, 2)])

% for t = 1:T
% %     plot(P3_T(t, 1), P3_T(t, 2), 'o', 'Color', cmap(t,:), 'MarkerFaceColor', cmap(t,:));
%     if t > 1
%         plot([P3_T(t-1, 1) P3_T(t, 1)], [P3_T(t-1, 2) P3_T(t, 2)], '-', 'Color', cmap(t,:),  'LineWidth',2);
%     end
% end
% arrow([P3_T(T-1, 1) P3_T(T-1, 2)], [P3_T(T, 1) P3_T(T, 2)])

% colorbar; % 显示颜色条
% colormap(jet);
% title('收敛过程（颜色表示迭代顺序）');
% xlabel('x'); ylabel('y');
% grid on



% m =20;n =10;p =4;
% A =randn(m,n);b =randn(m,1);
% C =randn(p,n);d =randn(p,1);e =rand;
% cvx_begin
%     variable x(n)
%     minimize(norm(A *x -b,2) )
%     subject to
%         C*x == d;
%         norm(x,Inf ) <= e;
% cvx_end
