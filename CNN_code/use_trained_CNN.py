import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from mynet import MyNet

exp_dict = {
    1: ['Sample1', 0.0003,   0.002, 0.012,  0],
    2: ['Sample2', 0.0003, 0.00175, 0.011, 30],
    3: ['Sample3', 0.0003, 0.00075, 0.009,  0],
    4: ['Sample4', 0.0003, 0.00125, 0.014, 45],
    5: ['Sample5', 0.0003, 0.00175, 0.011,  0],
    6: ['Sample6', 0.0003, 0.00125, 0.014, 15],
    7: ['Sample1_bot', 0.0003,   0.002, 0.007,  0],
    8: ['Sample2_bot', 0.0003, 0.00175, 0.008, 30],
    9: ['Sample3_bot', 0.0003, 0.00075, 0.01,  0],
    #10: ['Sample4_bot', 0.0003, 0.00125, 0.005, 45],
    10: ['Sample5_bot', 0.0003, 0.00175, 0.008,  0],
    #12: ['Sample6_bot', 0.0003, 0.00125, 0.005, 15],
    11: ['top_left', 0.0003, 0.0025, 0.01, 30],
    15: ['bot_left', 0.0003, 0.0025, 0.009, 30],
    12: ['top_mid', 0.0003, 0.0015, 0.007, 15],
    14: ['bot_mid', 0.0003, 0.0015, 0.012, 15],
    13: ['top_right', 0.0003, 0.00175, 0.009, 60],
    16: ['bot_right', 0.0003, 0.00175, 0.01, 60],
    17: ['Sample14', 0.0003, 0.002, 0.01, 45],
    18: ['Sample14_bot', 0.0003, 0.002, 0.009, 45],
    19: ['Sample15', 0.0003, 0.0015, 0.013, 60],
    #20: ['Sample15_bot', 0.0003, 0.0015, 0.006, 60],
    20: ['Sample16', 0.0003, 0.001, 0.008, 75],
    21: ['Sample16_bot', 0.0003, 0.001, 0.011, 75]
}

test_dataset_dict = {
    3: 'Size_Depth',
    4: 'Orientation',
    5: 'Size_Orientation',
    6: 'Size_Orientation_randloc'
}

# Choose samples
sample = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

# Choose a testing dataset
test = 0
dataset = 6

# Manual flag
target_flags = [0, 1, 1, 1]

# Load experimental data
signals_exp = np.ndarray([0,4001])
for i in sample:
    temp = np.expand_dims(np.genfromtxt('exp_final_model5/' + exp_dict[i][0] + '.txt',
                            delimiter=',', dtype=np.float32), axis=0)
    signals_exp = np.concatenate([signals_exp, temp])

targets_exp = np.ndarray([4, 0])
for i in sample:
    temp = np.expand_dims(np.array(exp_dict[i][1:5]).transpose(), axis=1)
    targets_exp = np.concatenate([targets_exp, temp], axis=1)
targets_exp = targets_exp[1:4, :]

# Convert to tensor
inputs_exp = torch.unsqueeze(torch.from_numpy(signals_exp[:, 601:]/1.043936254833034e-09).float(), 1)
targets_exp = torch.tensor(targets_exp).float()

if targets_exp.ndim == 1:
    targets_exp = torch.unsqueeze(targets_exp, 1)
else:
    targets_exp = torch.transpose(targets_exp, 0, 1)

if test:
    # Load test data
    signals_test = pd.read_excel('test/' + test_dataset_dict[dataset] + '.xlsx',
                                 header=None).values
    signals_test_info = pd.read_excel('test/' + test_dataset_dict[dataset] + '_info.xlsx',
                                      header=None).values
    signals_test = signals_test[~np.isnan(signals_test).any(axis=1)]
    signals_test_info = signals_test_info[:, ~np.isnan(signals_test_info).any(axis=0)]
    targets_test = signals_test_info[1:4, :]

    inputs_test = torch.unsqueeze(torch.from_numpy(abs(signals_test[:, 601:]/1.043936254833034e-09)).float(), 1)
    targets_test = torch.from_numpy(targets_test).float()

    if targets_exp.ndim == 1:
        targets_test = torch.unsqueeze(targets_test, 1)
    else:
        targets_test = torch.transpose(targets_test, 0, 1)

# Load the trained CNN
# model3 is currently used in the 3D paper
model = torch.load('trained_model/model5.pt')

model.eval()
with torch.no_grad():
    outputs_exp = model(inputs_exp)

outputs_exp = outputs_exp.numpy()
targets_exp = targets_exp.numpy()

v_range = np.array([[0.0003, 0.0003],  # short axis
                    [0.0005, 0.0025],  # size
                    [0.007, 0.015],  # depth
                    [0, 90]])  # orientation
column_index = 0
for i in range(4):
    if target_flags[i]:
        outputs_exp[:, column_index] = outputs_exp[:, column_index] * (
                    v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        column_index += 1

exp_error = abs((outputs_exp - targets_exp) / targets_exp) * 100
print('Experimental predictions are:')
for item in outputs_exp.tolist():
    item[0] = item[0] * 2
    print(item)
print('Experimental true values are:')
for item in targets_exp.tolist():
    item[0] = item[0] * 2
    print(item)
print('Experimental errors are:')
for item in exp_error.tolist():
    print(item)

# Calculate MAPE and MAE for experimental data
error = targets_exp - outputs_exp
abs_error = abs(error)
rel_error = error / targets_exp * 100
abs_rel_error = abs(rel_error)
MAPE = np.sum(abs_rel_error, 0) / abs_rel_error.shape[0]
MAE = np.sum(abs_error, 0) / abs_error.shape[0]
MAE[:2] = MAE[:2] * 1000
print('MAPE for experiment data is', MAPE)
print('MAE for experiment data is', MAE)

# Write the variables to file and process with MATLAB for paper quality figures
'''
wb1 = xlsxwriter.Workbook('../../MATLAB/NDT/3D MATLAB files/CNN_results_exp.xlsx')
sheet1 = wb1.add_worksheet()  # writes exp data prediction
sheet2 = wb1.add_worksheet()  # writes exp data actual value

for i in range(outputs_exp.shape[0]):
    for j in range(outputs_exp.shape[1]):
        sheet1.write_number(i, j, outputs_exp[i][j])
        sheet2.write_number(i, j, targets_exp[i][j])
wb1.close()'''

if test:
    with torch.no_grad():
        outputs_test = model(inputs_test)

    outputs_test = outputs_test.detach().numpy()
    targets_test = targets_test.numpy()

    for i in range(4):
        if target_flags[i]:
            outputs_test[:, column_index] = outputs_test[:, column_index] * (
                    v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
            column_index += 1

    # Calculate MAPE for test data
    abs_error = targets_test - outputs_test
    rel_error = abs_error / targets_test * 100
    abs_rel_error = abs(rel_error)
    MAPE = np.sum(abs_rel_error, 0) / abs_rel_error.shape[0]
    print('MAPE for test data is', MAPE)

    # A rolling index to indicate which output to be plotted
    plot_index = 0

    if target_flags[1]:
        # Plot P-A for size
        PAline = np.linspace(0.5, 2.5, num=30)
        fig, ax = plt.subplots()
        ax.scatter(targets_test[:, plot_index] * 1000, outputs_test[:, plot_index] * 1000,
                   s=150, marker='s', color='red')
        ax.plot(PAline, PAline, 'k--', linewidth=5)
        ax.set_xlabel('Actual Value (mm)', fontsize=30)
        ax.set_ylabel('Predicted Value (mm)', fontsize=30)
        ax.set_xticks(np.arange(0.5, 2.5))
        ax.set_yticks(np.arange(0.5, 2.5))
        ax.set_xlim([0.5, 2.5])
        ax.set_ylim([0.5, 2.5])
        ax.tick_params(labelsize=25)
        ax.set_aspect('equal', adjustable='box')
        fig.set_size_inches(8, 8)
        plt.title('Crack size prediction', fontsize=30)
        plt.legend(['Pred=Actual', 'Testing data'], fontsize=22, frameon=False)
        # plt.savefig('Size_pred.png', format='png', dpi=600)
        plt.show()
        plot_index += 1

    if target_flags[2]:
        # Plot P-A for depth
        PAline = np.linspace(7, 15, num=30)
        fig, ax = plt.subplots()
        ax.scatter(targets_test[:, plot_index] * 1000, outputs_test[:, plot_index] * 1000,
                   s=150, marker='s', color='red')
        ax.plot(PAline, PAline, 'k--', linewidth=5)
        ax.set_xlabel('Actual Value (mm)', fontsize=30)
        ax.set_ylabel('Predicted Value (mm)', fontsize=30)
        ax.set_xticks(np.arange(7, 16))
        ax.set_yticks(np.arange(7, 16))
        ax.set_xlim([7, 15])
        ax.set_ylim([7, 15])
        ax.tick_params(labelsize=25)
        ax.set_aspect('equal', adjustable='box')
        fig.set_size_inches(8, 8)
        plt.title('Crack depth prediction', fontsize=30)
        plt.legend(['Pred=Actual', 'Testing data'], fontsize=22, frameon=False)
        # plt.savefig('Depth_pred.png', format='png', dpi=600)
        plt.show()
        plot_index += 1

    if target_flags[3]:
        # Plot P-A for orientation
        PAline = np.linspace(0, 90, num=30)
        fig, ax = plt.subplots()
        ax.scatter(targets_test[:, plot_index], outputs_test[:, plot_index],
                   s=150, marker='s', color='red')
        ax.plot(PAline, PAline, 'k--', linewidth=5)
        ax.set_xlabel('Actual Value', fontsize=30)
        ax.set_ylabel('Predicted Value', fontsize=30)
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 90])
        ax.tick_params(labelsize=25)
        ax.set_aspect('equal', adjustable='box')
        fig.set_size_inches(8, 8)
        plt.title('Crack Orientation prediction', fontsize=30)
        plt.legend(['Pred=Actual', 'Testing data'], fontsize=22, frameon=False)
        # plt.savefig('Orientation_pred.png', format='png', dpi=600)
        plt.show()

