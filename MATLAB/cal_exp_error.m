%%% A script to calculate experimental signal error in different fashions
clc
clear all
close all

exp_actual = [90, 60, 90, 45, 90, 75, 90, 60, 90, 90, 60, 75, ...
    30, 75, 60, 30];
exp_pred = [79.93, 66.77, 57.58, 45.12, 91.90, 73.19, 79.78, 63.55, ...
    71.23, 90.27, 64.96, 76.57, 40.07, 81.34, 61.65, 43.28];
exp_error = exp_pred - exp_actual;
exp_absper_error = abs(exp_error / 90) * 100;
exp_ave_error = sum(exp_absper_error) / length(exp_actual);
