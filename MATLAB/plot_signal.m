%%% Add dummy data point in experimental signal to match simulation %%%
clc
clear all
close all

experiment_data = xlsread('../Experimental_signal/Sample3.xlsx')';
simulation_data1 = xlsread('../Experimental_signal/Sample5_sim.xlsx');

%%
experiment_data = experiment_data + 10;
time = [0:2e-9:8e-6];

% useful_len = length(experiment_data);
useful_len = 3200;
target_len = length(time);

%dummy = 2 + sort(randperm(target_len - 2, target_len - useful_len));
dummy = 10:floor(useful_len/(target_len - useful_len)):...
    (target_len - useful_len + 1)*floor(useful_len/(target_len - useful_len));

new_signal = linspace(0,0,target_len);
new_signal(dummy) = 1;

j = 1;
for i = 1:target_len
    if new_signal(i) == 0
        new_signal(i) = experiment_data(j);
        j = j + 1;
    else
        new_signal(i) = new_signal(i-1);
    end
end

experiment_data = experiment_data - 10;
new_signal = new_signal - 10;
% plot(experiment_data,'linewidth',3)
% hold on
% plot(new_signal,'linewidth',3)
% legend('Experiment','Simulation')
% set(gca,'FontSize',44)
% set(gca,'YColor','k')
% set(gca,'LineWidth',2);
% set(gcf,'Units','Inches');
% set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])
% legend('Original','Padded')
%%
simulation_data1(1:600) = 0;
new_signal(1:600) = 0;

figure
plot(new_signal(600:end)/max(new_signal),'k','linewidth',3)
xlim([0, 3400])
box on
% ylabel('Displacement (m)')
% xlabel('Time (s)')
set(gca,'FontSize',44)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])

figure
plot(abs(simulation_data1(600:end)/max(simulation_data1)),'k','linewidth',3)
box on
xlim([0, 3400])
% ylabel('Displacement (m)')
% xlabel('Time (s)')
set(gca,'FontSize',44)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])

%%
%exp_out = 'C:\Users\sniu3\Documents\python_work\ellip_3D_CNN\exp_test\';
%writematrix(new_signal/10^12, [exp_out,'Sample3.txt']);