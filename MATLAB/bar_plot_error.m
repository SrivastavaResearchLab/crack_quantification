%%% Plot bar plot for testing data and experimental data %%%
clc
clear all
close all

dummyx = [1];
mape = [5.0 5.7; 3.6 5.6; 4.9 8.4];
absolute1 = [0.08 0.08; 0.39 0.57];
absolute2 = [4.4; 7.5];

% Top bar graph
figure
subplot(2, 2, [1 2]);
bar(mape, 0.4);
set(gca,'FontSize',32)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 12 10.2])
grid on
grid minor
set(gca, 'MinorGridLineStyle', '--');
set(gca, 'XTick', []);

subplot(2, 2, 3);
bar(absolute1, 0.6)
set(gca,'FontSize',32)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 12 10.2])
grid on
grid minor
set(gca, 'MinorGridLineStyle', '--');
set(gca, 'XTick', []);

subplot(2, 2, 4);
bar(dummyx, absolute2, 0.4)
ylim([0, 10])
set(gca,'FontSize',32)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 12 10.2])
grid on
grid minor
set(gca, 'MinorGridLineStyle', '--');
set(gca, 'XTick', []);