clear all;
close all;
Loss = [[3776071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481];
    [4106432277.427634, 3653755103.2599854, 3984617028.24047, 4126651629.719391];
    [3819717104.7990537, 4100063794.523541, 3649757438.1855006, 4333580923.156441];
%     [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001];
    [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601];
    [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]];

% Loss = [[3176071953.5462384, 3685714309.525074, 3713654933.068316, 4826858139.903481];
%     [4106432277.427634, 3153755103.2599854, 3984617028.24047, 4126651629.719391];
%     [3819717104.7990537, 4100063794.523541, 3149757438.1855006, 4333580923.156441];
% %     [4329283766.372562, 4068219975.751717, 3920903631.1161523, 3632477833.380001];
%     [3751949385.2653327, 3569275009.1528945, 3749556634.9107757, 4377347096.690601];
%     [3975061010.9144645, 3984617028.24047, 3685714309.525074, 3883782305.5850515]];
env = struct();
env.P_max = 1;  % max transmission power
env.Bandwidth = 0.5e6;  % Hz
env.n0 = -170;  % dBm/Hz 计算得到的白噪声功率近似为0
env.sigma2 = 10 .^ (env.n0 * env.Bandwidth / 10) * 0.001;  % W
env.a = 11.95;
env.b = 0.14;
env.uNLoS = 23;
env.uLoS = 3;  % dB
env.fc = 2000;  % carrier frequency Hz
env.c = 3e8;  % light speed m/s
env.H = 150;  % Height m
env.E = 0.3 * 0.5e6;  %
global c_f c_o;
c_f = 0.12 * 0.5e6; % follower 能耗系数 (线性)
c_o = 1.5 * 0.5e6;  % leader 能耗系数 (线性)
env.count = 0;      
Pmax_f = 3 .* [env.P_max;env.P_max;env.P_max];   % follower 最大功率
Pmax_o = 2 .* [env.P_max;env.P_max];     % leader 最大功率 

J = 2; M = 3;

%-----------------------定义变量的排布index，包括Po,P,KKT的参数
idx = struct();
idx.Po = 1:J;
idx.P = (J+1):(J+M);
idx.KKT_lambda = (J+M+1):(J+M+M);
idx.KKT_mu = (J+M+M+1):(J+M+M+M);
idx.KKT_theta = (J+3*M+1):(J+4*M);
tail = J+4*M;

%------------------------用于存储迭代过程数据
T = 500;
% Iteration = [1:T];
% res = struct();
% res.count = 0;
% res.PJ1_T = zeros(1, T);
% res.PJ2_T = zeros(1, T);
% 
% res.PN1_T = zeros(1, T);
% res.PN2_T = zeros(1, T);
% res.PN3_T = zeros(1, T);
%----------------------- 初始化所有变量
x0 = 2 * rand(tail, 1)
x0(1:5) = [

   1.0
    1.5010
    1.7309
    1.2251
    0.5

]
% x0 = [
% 
%    1.0
%     1.5010
%     1.7309
%     1.2251
%     0.5
%     1.0554
%     0.9590
%     1.6027
%     0.4557
%     0.9962
%     1.8017
%     1.1493
%     1.6904
%     1.4773
% ]
% x0 = zeros(tail, 1);
% x0(idx.Po) = 0.2;
% x0(idx.P) = 0.9;
% x0(idx.KKT_lambda) = 0.1;
% x0(idx.KKT_mu) = 0.1;
% x0(idx.KKT_theta) = 0.1;

%-----------------------限制变量的上下界
lb = zeros(tail,1); % 全部非负
ub = inf(tail,1);
ub(idx.Po) = 2 * env.P_max; % 功率上界
ub(idx.P) = 3 * env.P_max;


%------------------------设置fmincon options

options = optimoptions('fmincon',...
    'Display','iter','MaxFunctionEvaluations',1e5,'MaxIterations',T, 'PlotFcn', @myOutputFcn);
% options = optimoptions('fmincon','Algorithm','interior-point',...
%     'Display','iter-detailed','MaxFunctionEvaluations',1e5,'MaxIterations',1000);

%--------------------------目标函数
obj = @(x) leader_objective(x, idx);
c_non_line = @(x) followers_kkt_constraints(x, idx, env, Loss);

A = [];
b = [];
Aeq = [];
beq = [];
% ---------------------------执行fmincon求解
[x_opt, fval, exitflag, output] = fmincon(obj, x0, A, b, Aeq, beq, lb, ub, c_non_line, options);

% 展示结果
p_o_opt = x_opt(idx.Po);
p_f_opt = x_opt(idx.P);
disp('=== 结果 ===');
disp(['p_o = ', mat2str(p_o_opt',6)]);
disp(['p_f = ', mat2str(p_f_opt',6)]);
disp(['leader objective (u^o) = ', num2str(-fval)]); % fmincon 最小化，所以 -fval 为 leader 的效用

% ------------------------------绘制结果
% figure(1);
% hold on; grid on;
% plot(T, res.PJ1_T, 'LineWidth',2, 'Displayname','J1');


function val = leader_objective(x, idx)
    global c_f c_o;
    PN = x(idx.P)';
    PJ = x(idx.Po)';
    
    I = I_calcu(PN, PJ);
    [RN, uN] = uN_calcu_MPCC(PN, I);
    
    % uN = RN - c_f * PN
%     uo1 = -sum(uN) - c_f * sum(PN) - c_o * sum(PJ)
    uo = -sum(RN) - c_o * sum(PJ);
    val = -uo;
end



function [c_neq, c_eq] = followers_kkt_constraints(x, idx, env, Loss)
    global c_f;
    PN = x(idx.P);
    PJ = x(idx.Po);
    lambda = x(idx.KKT_lambda);
    mu = x(idx.KKT_mu);
    theta = x(idx.KKT_theta);
    
    c_neq = [];
    c_eq = [];
    % 不等式约束，-PN <0, PN -P_max < 0
    Pmax_f = 3 .* [env.P_max;env.P_max;env.P_max];              % 限制最大值
    c_neq = [c_neq; -PN; PN - Pmax_f; -lambda; -mu; -theta];
    I = I_calcu(PN', PJ');
%     [~, uN] = uN_calcu_MPCC(PN', I);
    N = length(PN);
    df_dp = [];
    for i = 1:N
        dfi_dpi = -env.Bandwidth / (PN(i) + Loss(i, i) * ( env.sigma2 + I(i))) + c_f;
        df_dp = [df_dp; dfi_dpi];
    end
    % 暂时只考虑一个上界，不考虑mu
    dL_dp = df_dp + lambda - theta;
    % 驻点，要求N个拉格朗日函数为0
    c_eq = [c_eq; dL_dp];
    for i = i:N
        c_eq = [c_eq; lambda(i) * (PN(i) - env.P_max)];
        c_eq = [c_eq; theta(i) * PN(i)];
    end
end


function stop = myOutputFcn(x, optimValues, state)
    stop = false; % 不停止优化过程
    persistent PN1; % 持久化变量，用于存储功率 历史
    persistent PN2; 
    persistent PN3; 
    persistent PJ1; 
    persistent PJ2; 
    persistent iters fvals feasi grads;
    persistent uN1 uN2 uN3 uJ;
    if isequal(state, 'init')
        PN1 = []; % 初始化历史记录为空
        PN2 = [];
        PN3 = [];
        PJ1 = [];
        PJ2 = [];
        iters = [];
        fvals = [];
        feasi = [];
        grads = [];
        uN1 = [];
        uN2 = [];
        uN3 = [];
        uJ = [];
    end
    if isequal(state, 'iter')
        % 在每次迭代时记录变量
        PJ1 = [PJ1, x(1)];
        PJ2 = [PJ2, x(2)];
        PN1 = [PN1, x(3)];
        PN2 = [PN2, x(4)];
        PN3 = [PN3, x(5)];
        I = I_calcu(x(3:5), x(1:2));
        [RN, uN] = uN_calcu_MPCC(x(3:5), I);
        uN1 = [uN1, uN(1)];
        uN2 = [uN2, uN(2)];
        uN3 = [uN3, uN(3)];
        
        uJ = [uJ, uJ_calculate(uN, x(3:5), x(1:2))];
        
        iters = [iters optimValues.iteration];
        fvals = [fvals optimValues.fval];
        feasi = [feasi optimValues.constrviolation];
        grads = [grads norm(optimValues.gradient)];
        
%         fprintf('Iteration %d: p_o = [%s], p_f = [%s]\n', optimValues.iteration, mat2str(p_o), mat2str(p_f));
    end
    if isequal(state, 'done')
        % 在优化结束时绘制 动作 的历史变化图
        figure(2);
        plot(1:length(PN1), PN1,  'LineWidth',1.2, 'LineStyle','-', 'Marker', '^', 'DisplayName', 'N1', 'MarkerIndices', 1:10:length(PN1)); hold on;
        plot(1:length(PN2), PN2,  'LineWidth',1.2, 'LineStyle','-.', 'Marker', 'd', 'DisplayName', 'N2', 'MarkerIndices', 1:10:length(PN2));
        plot(1:length(PN3), PN3,  'LineWidth',1.2, 'LineStyle',':', 'Marker', 's', 'DisplayName', 'N3', 'MarkerIndices', 1:10:length(PN3));
        
        plot(1:length(PJ1), PJ1,  'LineWidth',1.2, 'LineStyle','-', 'Marker', 'x', 'DisplayName', 'J1', 'MarkerIndices', 1:10:length(PJ1));
        plot(1:length(PJ2), PJ2,  'LineWidth',1.2, 'LineStyle','--', 'Marker', 'o', 'DisplayName', 'J2', 'MarkerIndices', 1:10:length(PJ2));

        xlabel('Iteration');
        ylabel('Power (W)');
        ylim([-0.2 3.2]);
        legend('location', 'best');
        grid on;
        % 绘制Feasibility和optimality
        figure(3);
        plot(iters,feasi,"o");hold on;
        plot(iters,grads,"x");
        xlabel('Iteration');
%         ylabel('Power (W)');
        
        % 绘制效用
        figure(4);
        plot(1:length(uN1), uN1,  'LineWidth',1.2, 'LineStyle','-', 'Marker', '^', 'DisplayName', 'N1', 'MarkerIndices', 1:10:length(PN1)); hold on;
        plot(1:length(uN2), uN2,  'LineWidth',1.2, 'LineStyle','-.', 'Marker', 'd', 'DisplayName', 'N2', 'MarkerIndices', 1:10:length(PN2));
        plot(1:length(uN3), uN3,  'LineWidth',1.2, 'LineStyle',':', 'Marker', 's', 'DisplayName', 'N3', 'MarkerIndices', 1:10:length(PN3));
        
        plot(1:length(uJ), uJ,  'LineWidth',1.2, 'LineStyle','-', 'Marker', 'x', 'DisplayName', 'Jammers', 'MarkerIndices', 1:10:length(uJ));
        
        xlabel('Iteration');
        ylabel('Utility');
        legend('location', 'best');
        grid on;
    end
end

% 
% function mpcc_leader_followers_example
% % MPCC 示例：leaders 决策变量是 P^o (1x2)，followers p (1x3)
% % 使用 fmincon，把 followers 的 KKT 条件作为非线性约束
% 
% %% 参数设置（你可以根据实际情况改这些）
% % 通道增益（示例）
% g_diag = [2.0; 1.8; 1.5];          % g_{11}, g_{22}, g_{33} (own gains)
% G_inter = [ 0, 0.2, 0.1;           % g_{12}, g_{13}...
%             0.1, 0, 0.25;
%             0.05,0.15,0 ];
% % leader -> follower 干扰增益 h_{oi,k} (3x2)
% H_ol = [0.3, 0.25;
%         0.2, 0.15;
%         0.1, 0.05];
% 
% sigma2 = 1e-3;      % 噪声方差
% c_f = 0.1;          % follower 能耗系数 (线性)
% c_o = 0.2;          % leader 能耗系数 (线性)
% 
% Pmax_f = [1;1;1];   % follower 最大功率
% Pmax_o = [2;2];     % leader 最大功率
% 
% %% 决策变量向量 x 的排布
% % x = [ p_o(1:2); p_f(1:3); lambda_lo(1:3); lambda_up(1:3) ]
% n_o = 2; n_f = 3;
% idx = struct();
% idx.oo = 1:n_o;
% idx.ff = (n_o+1):(n_o+n_f);
% idx.l_lo = (n_o+n_f+1):(n_o+n_f+n_f);
% idx.l_up = (n_o+n_f+n_f+1):(n_o+n_f+n_f+n_f);
% nvar = idx.l_up(end);
% 
% %% 初始值
% x0 = zeros(nvar,1);
% x0(idx.oo) = 0.5;           % leaders 起始功率
% x0(idx.ff) = 0.2*ones(n_f,1);
% x0(idx.l_lo) = 0.1*ones(n_f,1);
% x0(idx.l_up) = 0.1*ones(n_f,1);
% 
% %% bounds (用 fmincon 的 lb/ub 限制 leaders 的界)
% lb = -inf(nvar,1);
% ub = inf(nvar,1);
% % leaders bounds:
% lb(idx.oo) = 0;
% ub(idx.oo) = Pmax_o;
% % followers bounds will be enforced inside nonlcon (作为不等式)
% % multipliers 非负：
% lb(idx.l_lo) = 0;
% lb(idx.l_up) = 0;
% 
% %% fmincon options
% options = optimoptions('fmincon','Algorithm','interior-point',...
%     'Display','iter','MaxFunctionEvaluations',1e5,'MaxIterations',1000);
% 
% % objective wrapper
% obj = @(x) leader_objective(x, idx, g_diag, G_inter, H_ol, sigma2, c_o);
% 
% % nonlinear constraints wrapper (包含 followers 的 KKT)
% nonlcon = @(x) followers_kkt_constraints(x, idx, g_diag, G_inter, H_ol, sigma2, c_f, Pmax_f);
% 
% % 无线性约束
% A = []; b = []; Aeq = []; beq = [];
% 
% % 求解
% [x_opt, fval, exitflag, output] = fmincon(obj, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
% 
% % 展示结果
% p_o_opt = x_opt(idx.oo);
% p_f_opt = x_opt(idx.ff);
% disp('=== 结果 ===');
% disp(['p_o = ', mat2str(p_o_opt',6)]);
% disp(['p_f = ', mat2str(p_f_opt',6)]);
% disp(['leader objective (u^o) = ', num2str(-fval)]); % fmincon 最小化，所以 -fval 为 leader 的效用
% 
% end
% 
% %% ---------------------- 辅助函数 -------------------------
% 
% function val = leader_objective(x, idx, g_diag, G_inter, H_ol, sigma2, c_o)
% % 目标： u^o = - sum_i R_i(p) - c_o * sum p_o
% % fmincon 做最小化，所以返回 -u^o = sum R_i + c_o * sum p_o
% p_o = x(idx.oo);
% p_f = x(idx.ff);
% 
% % 计算每个 follower 的 SINR 和 R_i
% R = zeros(length(p_f),1);
% for i=1:length(p_f)
%     num = g_diag(i) * p_f(i);
%     interf = 0;
%     for j=1:length(p_f)
%         if j~=i
%             interf = interf + G_inter(j,i) * p_f(j);
%         end
%     end
%     % leaders 对 follower 的干扰
%     interf_o = H_ol(i,:) * p_o(:);
%     denom = interf + interf_o + sigma2;
%     SINR = num / denom;
%     R(i) = log2(1 + SINR);
% end
% 
% u_o = - sum(R) - c_o * sum(p_o);    % leader 的效用
% val = -u_o; % fmincon 最小化 -> 返回 -u_o
% end
% 
% function [c, ceq] = followers_kkt_constraints(x, idx, g_diag, G_inter, H_ol, sigma2, c_f, Pmax_f)
% % 返回 fmincon 的非线性约束：
% % c(x) <= 0 （不等式）
% % ceq(x) == 0 （等式）
% %
% % 包括：
% % - follower 的箱约束 0 <= p_i <= Pmax_i (作为 c)
% % - multipliers 非负性 (在外面用 lb 已经设置 lambda >=0，但这里再写一次以稳健)
% % - 驻点方程 (stationarity) (等式)
% % - 互补乘积 p_i * lambda_lo_i = 0, (Pmax_i-p_i)*lambda_up_i = 0 (等式)
% 
% p_o = x(idx.oo);
% p_f = x(idx.ff);
% lam_lo = x(idx.l_lo);
% lam_up = x(idx.l_up);
% 
% n_f = length(p_f);
% 
% % 不等式约束 c(x) <= 0
% c = [];
% 
% % 1) followers 的箱约束 (写为 g(x) <= 0)
% c_box_lower = -p_f;             % -p_i <= 0  => p_i >= 0
% c_box_upper = p_f - Pmax_f;     % p_i - Pmax <= 0
% c = [c; c_box_lower; c_box_upper];
% 
% % 2) multipliers 非负 (虽然 lb 已设置，但重复写更稳健)
% c = [c; -lam_lo; -lam_up];
% 
% % 等式约束 ceq(x) == 0
% ceq = [];
% 
% % 计算每个 follower 的 derivative dR_i/dp_i
% dR_dp = zeros(n_f,1);
% R = zeros(n_f,1);
% denom_vec = zeros(n_f,1);
% for i=1:n_f
%     interf = 0;
%     for j=1:n_f
%         if j~=i
%             interf = interf + G_inter(j,i) * p_f(j);
%         end
%     end
%     interf_o = H_ol(i,:) * p_o(:);
%     denom = interf + interf_o + sigma2;   % denominator not including own term
%     denom_vec(i) = denom;
%     SINR = (g_diag(i)*p_f(i)) / denom;
%     R(i) = log2(1 + SINR);
%     % dR/dp_i = (1/ln2) * g_ii / (denom + g_ii*p_i)
%     dR_dp(i) = (1/log(2)) * g_diag(i) / (denom + g_diag(i)*p_f(i));
% end
% 
% % 驻点方程: derivative of follower objective w.r.t p_i:
% % df_i/dp_i = dR/dp_i - c_f
% % stationarity: df_i/dp_i - lambda_lo_i + lambda_up_i = 0
% stationarity = (dR_dp - c_f) - lam_lo + lam_up;
% ceq = [ceq; stationarity];
% 
% % 互补条件（等式）
% for i=1:n_f
%     ceq = [ceq; lam_lo(i) * p_f(i)];                % lambda_lo * p_i = 0
%     ceq = [ceq; lam_up(i) * (Pmax_f(i) - p_f(i))];  % lambda_up * (Pmax - p_i) = 0
% end
% 
% end
