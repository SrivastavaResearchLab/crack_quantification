clc
clear all
close all

% Load data
results = 'CNN_results_exp';

exp_pred = xlsread(results,'Sheet1');
exp_actual = xlsread(results,'Sheet2');

% Change orientation from 0 to 90
exp_pred(:,3) = 90 - exp_pred(:,3);
exp_actual(:,3) = 90 - exp_actual(:,3);

exp_error = exp_actual - exp_pred;
exp_abs_error = abs(exp_error);
exp_rel_error = exp_error ./ exp_actual * 100;
exp_rel_error(:,3) = exp_error(:,3) /90 * 100;
exp_absrel_error = abs(exp_rel_error);
exp_mape = sum(exp_absrel_error, 1) / length(exp_absrel_error);
exp_mae = sum(exp_abs_error, 1) / length(exp_abs_error);
exp_mae(1:2) = exp_mae(1:2) * 1000;

% Set to 1 if plot separately and set to 2 if using 3D plot
plot_style = 1;

if plot_style == 1
    figure
    hold on
    box on
    plot(exp_actual(:,1)*2000,exp_pred(:,1)*2000,'^','linewidth',15,'markersize',15,'color',[0.7, 0.3, 0.2])
    plot(linspace(1,5,100),linspace(1,5,100),'--k','linewidth',8);
    set(gca,'FontSize',44)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Experimental data','Prediction = Actual', 'location', 'northwest')
    xlim([1, 5])
    ylim([1, 5])

    figure
    hold on
    box on
    plot(exp_actual(:,2)*1000,exp_pred(:,2)*1000,'^','linewidth',15,'markersize',15,'color',[0.1, 0.3, 0.7])
    plot(linspace(7,15,100),linspace(7,15,100),'--k','linewidth',8);
    set(gca,'FontSize',44)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Experimental data','Prediction = Actual', 'location', 'northwest')
    xlim([7, 15])
    ylim([7, 15])
    xticks([7,9,11,13,15])
    yticks([7,9,11,13,15])
    
    figure
    hold on
    box on
    plot(exp_actual(:,3),exp_pred(:,3),'^','linewidth',15,'markersize',15,'color',[0.1, 0.7, 0.3])
    plot(linspace(0,90,100),linspace(0,90,100),'--k','linewidth',8);
    set(gca,'FontSize',44)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
    xlabel(['Actual Value (' char(176) ')'])
    ylabel(['CNN Prediction (' char(176) ')'])
    legend('Experimental data','Prediction = Actual', 'location', 'northwest')
    xlim([0, 90])
    ylim([0, 90])
    xticks([0 30, 60, 90])
    yticks([0 30, 60, 90])
end

if plot_style == 2
    figure
    plot3(data_exp(:,1),data_exp(:,3),data_exp(:,5),'bo'...
        ,'linewidth',1,'markersize',20,'markerfacecolor','b')
    hold on
    plot3(data_exp(:,2),data_exp(:,4),data_exp(:,6),'ro'...
        ,'linewidth',1,'markersize',20,'markerfacecolor','r')
    set(gca,'FontSize',44)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])
    xlabel('Size (mm)')
    ylabel('Location (mm)')
    zlabel('Orientation')
    legend('Actual','CNN predicted')
    box on
    grid on
end