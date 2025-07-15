import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from datetime import datetime
import re
import string

# --- Setup Paths and Plotting Style ---
folder_path = r'D:\GTIIT\paper\K2SO4Ultrasound\PotassiumSulfate\verification\turbidity'
save_path = r'D:\GTIIT\paper\K2SO4Ultrasound\PotassiumSulfate\verification\turbidity'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
c = sns.color_palette("deep")
letters = list(string.ascii_uppercase) 

# --- Helper Function to Sort Files ---
def extract_suffix_number(filename):
    """Extracts the numerical suffix from a filename for sorting."""
    match = re.search(r'_([\d.]+)(?=\.csv$)', filename)
    return float(match.group(1)) if match else float('inf')

# --- Load and Process Data ---
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort(key=extract_suffix_number, reverse=True)

dataframes = {}
for file in csv_files:
    csv_file = os.path.join(folder_path, file)
    df = pd.read_csv(csv_file)
    df['Time'] = pd.to_datetime(df['Time'])
    start_time = df['Time'].iloc[0]
    df['Time_elapsed'] = (df['Time'] - start_time).dt.total_seconds() / 60
    filename_no_ext = os.path.splitext(file)[0]
    dataframes[f'df_{filename_no_ext}'] = df

# --- Plotting ---
num_dfs = len(dataframes)
if num_dfs > 0:
    fig, ax = plt.subplots(num_dfs, 3, figsize=(15, 5 * num_dfs), sharex=False, squeeze=False)

    for i, (key, df) in enumerate(dataframes.items()):
        rate_match = re.search(r'_([\d.]+)$', key ) # $ 表示字符串结尾
        rate = float(rate_match.group(1)) if rate_match else None
        # # 1. Plot Temperature
        # ax[i, 0].plot(df['Time_elapsed'], df['Temperature'], color=c[0],label='Temperature')
        # # ax[i, 0].set_title(f'{key} - Temperature')
        # # ax[i, 0].set_xlabel('Time elapsed (minutes)')
        # ax[i, 0].set_ylabel('Temperature (°C)')
        # ax[i, 0].grid(True, linestyle='--')
# 画主轴：温度
        temp_line, = ax[i, 0].plot(df['Time_elapsed'], df['Temperature'], color=c[0], label='Temperature')
        ax[i, 0].set_title(f'({letters[i * 3 + 0]}) {rate} $^\circ$C/min - Temperature & Turbidity')
        ax[i, 0].set_xlabel('Time (min)')
        ax[i, 0].set_ylabel('Temperature (°C)', color=c[0])
        ax[i, 0].tick_params(axis='y', labelcolor=c[0])
        ax[i, 0].grid(True, linestyle='--')

        # 画副轴：浊度
        ax2 = ax[i, 0].twinx()
        turb_line, = ax2.plot(df['Time_elapsed'], df['Turbidity'], color=c[1], label='Turbidity')
        ax2.set_ylabel('Turbidity', color=c[1])
        ax2.tick_params(axis='y', labelcolor=c[1])

        # 合并 legend
        lines = [temp_line, turb_line]
        labels = [line.get_label() for line in lines]
        ax[i, 0].legend(lines, labels, loc='center left')  # 或 loc='best'


        # 2. Plot Concentration, Solubility, and MSZW
        ax[i, 1].plot(df['Time_elapsed'], df['Concentration'], label='Concentration', color=c[1])
        ax[i, 1].plot(df['Time_elapsed'], df['Solubility (g/L)'], label='Solubility', color=c[2])


        # 设置对应速率下的成核温度 threshold_temp
        threshold_temp = None
        if rate == 0.5:
            threshold_temp = 28
        elif rate == 0.3:
            threshold_temp = 30
        elif rate == 0.1:
            threshold_temp = 33
        # print(rate)
        # print(threshold_temp)
        # 获取成核时间（温度首次低于 threshold_temp 的时间）
        if threshold_temp is not None:
            below_threshold = df[df['Temperature'] <= threshold_temp]
            if not below_threshold.empty:
                t_nucleation = below_threshold['Time_elapsed'].iloc[0]
            else:
                t_nucleation = df['Time_elapsed'].max()
        else:
            t_nucleation = df['Time_elapsed'].max()

        # 创建布尔掩码：过饱和 + 时间小于成核时间
        conc_mask = df['Concentration'] > df['Solubility (g/L)']
        time_mask = df['Time_elapsed'] <= t_nucleation
        metastable_mask = conc_mask & time_mask  # 是一个和 df 一样长的 Boolean Series

        # 填充图中灰色区域表示 MSZW
        ax[i, 1].fill_between(df['Time_elapsed'],
                            df['Concentration'],
                            df['Solubility (g/L)'],
                            where=metastable_mask,
                            color='gray',
                            alpha=0.5,
                            label='MSZW')

        
        ax[i, 1].set_title(f'({letters[i * 3 + 1]}) {rate} $^\circ$C/min - Concentration vs Solubility')
        ax[i, 1].set_xlabel('Time (min)')
        ax[i, 1].set_ylabel('g/L')
        ax[i, 1].grid(True, linestyle='--')
        ax[i, 1].legend()

        # 3. Plot Supersaturation
        ax[i, 2].plot(df['Time_elapsed'], df['Supersaturation_ln'], color=c[3])
        ax[i, 2].set_title(f'({letters[i * 3 + 2]}) {rate} $^\circ$C/min - Supersaturation')
        ax[i, 2].set_xlabel('Time (min)')
        ax[i, 2].set_ylabel('Supersaturation')
        ax[i, 2].grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig('verification.eps')
    plt.savefig('vertification.pdf')
    plt.show()

else:
    print(f"No CSV files found in the directory: {folder_path}")