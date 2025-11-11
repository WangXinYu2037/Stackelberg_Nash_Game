clear,clc
t = [0:0.01:20];
t = t.';
x = t.*cos(t);
y = t.*sin(t);
y2 = t.^2 .* sin(t);
arrowPlot(x, y, 'number', 5, 'color', 'r', 'LineWidth', 1)
plot_star(5, 5);

% x = rand(10, 1);
% y = rand(10,1);
% cmap = jet(length(x)); % 生成颜色
% hold on
% for k = 1:length(x)
%     plot(x(k), y(k), 'o', 'Color', cmap(k,:), 'MarkerFaceColor', cmap(k,:));
%     if k > 1
%         plot([x(k-1) x(k)], [y(k-1) y(k)], '-', 'Color', cmap(k,:));
%     end
% end
% colorbar; % 显示颜色条
% colormap(jet);
% title('收敛过程（颜色表示迭代顺序）');
% xlabel('x'); ylabel('y');
% grid on
% x = rand(10,2);
% plot(x(:, 1), x(:, 2))