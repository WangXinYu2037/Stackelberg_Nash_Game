x = 0:0.01:0.2;
y1 = 7 + 3 * tanh(x);
y2 = 7 + 4 * tanh(x);

% Main plot
figure;
plot(x, y1, 'bo-', 'DisplayName', 'MRT');
hold on;
plot(x, y2, 'r-s', 'DisplayName', 'ZF');
xlabel('Leakage Threshold \delta');
ylabel('Average Min Rate (bps/Hz)');
legend;


% Inset axes
ax1 = axes('Position', [0.2 0.5 0.3 0.3]); % Position of inset: [left bottom width height]
hold on;
plot(ax1, x, y1, 'bo-', 'DisplayName', 'MRT');
plot(ax1, x, y2, 'r-s', 'DisplayName', 'ZF');

% Adjust axis limits for zoom effect
xlim([0.085 0.1]);
ylim([7.2 7.5]);
set(ax1, 'Box', 'on'); % Box around inse