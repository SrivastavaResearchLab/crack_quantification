%%% Plot for paper on 3D NDT crack quantification %%%
clc
clear all
close all

% Load data
results = 'CNN_results_model5'; % Use _aug for the time being

test_pred = xlsread(results,'Sheet3');
test_actual = xlsread(results,'Sheet4');

% Change orientation from 0 to 90
test_pred(:,3) = 90 - test_pred(:,3);
test_actual(:,3) = 90 - test_actual(:,3);

%%
% Plot switches
Loss = 1;
Size = 1;
Location = 1;
Orientation = 1;
Size_and_Orientation = 0;

test_error = test_actual - test_pred;
test_abs_error = abs(test_error);
test_rel_error = test_error ./ test_actual * 100;
test_rel_error(:,3) = test_error(:,3) /90 * 100;
test_absrel_error = abs(test_rel_error);
test_mape = sum(test_absrel_error, 1) / length(test_absrel_error);
test_mae = sum(test_abs_error, 1) / length(test_abs_error);
test_mae(1:2) = test_mae(1:2) * 1000;

% exp_error = exp_actual - exp_pred;
% exp_abs_error = abs(exp_error);
% exp_rel_error = exp_error ./ exp_actual * 100;
% exp_absrel_error = abs(exp_rel_error);
% exp_mape = sum(exp_absrel_error, 1) / length(exp_absrel_error);
% exp_mae = sum(exp_abs_error, 1) / length(exp_abs_error);
% exp_mae(1:2) = exp_mae(1:2) * 1000;

%%%%%%%%%%%%%%%%%%%%%%%%% Plot size %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Size
    figure
    hold on
    plot(test_actual(:,1)*2000,test_pred(:,1)*2000,'^','linewidth',7,'markersize',7,'color',[0.7, 0.3, 0.2]);
    plot(linspace(1,5,100),linspace(1,5,100),'--k','linewidth',5);
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Testing data','Prediction = Actual', 'location', 'northwest')
    xlim([1, 5])
    ylim([1, 5])
    box on
    set(gca,'FontSize',40)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
end

%%%%%%%%%%%%%%%%%%%%%%%% Plot location %%%%%%%%%%%%%%%%%%%%%%%%%%%
if Location
    figure
    hold on
    plot(test_actual(:,2)*1000,test_pred(:,2)*1000,'^','linewidth',7,'markersize',7,'color',[0.1, 0.3, 0.7]);
    plot(linspace(7,15,100),linspace(7,15,100),'--k','linewidth',5);
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Testing data','Prediction = Actual', 'location', 'northwest')
    xlim([7, 15])
    ylim([7, 15])
    xticks([7,9,11,13,15])
    yticks([7,9,11,13,15])
    box on
    set(gca,'FontSize',40)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
end

%%%%%%%%%%%%%%%%%%%%%% Plot orientation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Orientation
    figure
    hold on
    plot(test_actual(:,3),test_pred(:,3),'^','linewidth',7,'markersize',7, 'color',[0.1, 0.7, 0.3]);
    plot(linspace(0,90,100),linspace(0,90,100),'--k','linewidth',8)
    xlabel(['Actual Value (' char(176) ')'])
    ylabel(['CNN Prediction (' char(176) ')'])
    legend('Testing data','Prediction = Actual', 'location', 'northwest')
    xlim([0, 90])
    ylim([0, 90])
    xticks([0, 30, 60, 90])
    yticks([0, 30, 60, 90])
    box on
    set(gca,'FontSize',40)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
end

%%%%%%%%%%%%%%%%%%% Plot size and orientation together %%%%%%%%%%%%%%%%%
if Size_and_Orientation
    figure
    hold on
    fig = gcf;
    fig.Units = 'Inches';
    fig.Position = [2 0.2 12 10.2];
    % fig.OuterPosition = [1 0 1.3*10. 1.5*7.5];
    p1 = plot(test_actual(:,1)*2000,test_pred(:,1)*2000,'s','color','#D95319','linewidth',2,'markersize',20);
    ax1 = gca;
    ax1.FontSize = 44;
    ax1.XAxis.Color = '#D95319';
    ax1.YAxis.Color = '#D95319';
    ax1.LineWidth = 3;
    ax1.YLim = [1,5];
    ax1.XLim = [1,5];
    ax1.XLabel.String = ('Actual Size (mm)');
    ax1.YLabel.String = ('CNN Predicted Size (mm)');
    ax1.XLabel.Color = '#D95319';
    ax1.YLabel.Color = '#D95319';
    ax1.DataAspectRatio = [1 1 1];
    ax1_pos = ax1.Position; % position of first axes
    ax2 = axes('Position',ax1_pos,...
        'XAxisLocation','top',...
        'YAxisLocation','right',...
        'Color','none');
    hold on
    p2 = plot(test_actual(:,3),test_pred(:,3),'^','color','#0072BD',...
        'parent',ax2,'linewidth',2,'markersize',20);
    p3 = plot(linspace(0,90,100),linspace(0,90,100),'--k','linewidth',8);
    ax2.FontSize = 44;
    ax2.XColor = '#0072BD';
    ax2.YColor = '#0072BD';
    ax2.LineWidth = 3;
    ax2.YLim = [0,90];
    ax2.XLim = [0,90];
    ax2.XLabel.String = ('Actual Orientation');
    ax2.YLabel.String = ('CNN Predicted Orientation');
    ax2.XTick = [0 30 60 90];
    ax2.YTick = [0 30 60 90];
    ax2.XLabel.Color = '#0072BD';
    ax2.YLabel.Color = '#0072BD';
    ax2.DataAspectRatio = [1 1 1];
end