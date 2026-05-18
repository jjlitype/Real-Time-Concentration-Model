import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from datetime import datetime
import re
import string
from scipy.signal import savgol_filter

# --- Setup Paths and Plotting Style ---
folder_path = '/home/kemove/WorkpaceP2/junjie/K2SO4Ultrasound/PotassiumSulfate/verification/turbidity'
save_path = '/home/kemove/WorkpaceP2/junjie/K2SO4Ultrasound/PotassiumSulfate/verification/turbidity'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
c = sns.color_palette("deep")
letters = list(string.ascii_uppercase) 

# --- Helper Function to Sort Files ---
def extract_suffix_number(filename):
    """Extracts the numerical suffix from a filename for sorting."""
    match = re.search(r'_([\d.]+)(?=\.csv$)', filename)
    return float(match.group(1)) if match else float('inf')

# --- Savitzky-Golay Smoothing Function ---
def apply_savgol_filter(data, window_size=61, poly_order=3, reflect_size=30):
    """
    Apply Savitzky-Golay filter with reflection at boundaries to mitigate edge effects.
    
    Parameters:
    - data: array-like, the input data series
    - window_size: int, the length of the filter window (must be odd)
    - poly_order: int, the order of the polynomial used to fit the samples
    - reflect_size: int, number of points to reflect at each end
    
    Returns:
    - smoothed_data: array, the filtered data
    """
    data_array = np.array(data)
    
    # Reflect the data at both ends
    left_reflect = data_array[:reflect_size][::-1]
    right_reflect = data_array[-reflect_size:][::-1]
    extended_data = np.concatenate([left_reflect, data_array, right_reflect])
    
    # Apply Savitzky-Golay filter
    smoothed_extended = savgol_filter(extended_data, window_size, poly_order)
    
    # Extract the middle portion (original data length)
    smoothed_data = smoothed_extended[reflect_size:-reflect_size]
    
    return smoothed_data

# --- Load and Process Data ---
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# 可选：排除已知的汇总/统计文件，避免列缺失报错
exclude_patterns = {'mszw_summary.csv'}
csv_files = [f for f in csv_files if f not in exclude_patterns]
csv_files.sort(key=extract_suffix_number, reverse=True)

dataframes = {}
required_cols = {'Time', 'Temperature', 'Concentration', 'Solubility (g/L)', 'Supersaturation_ln', 'Turbidity'}

for file in csv_files:
    csv_file = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取失败，跳过 {file}: {e}")
        continue

    missing = required_cols - set(df.columns)
    if missing:
        print(f"跳过 {file}，缺少列: {missing}")
        continue

    # 若 Time 已经是 datetime 则直接使用；否则转换
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        try:
            df['Time'] = pd.to_datetime(df['Time'])
        except Exception as e:
            print(f"无法解析 Time 列，跳过 {file}: {e}")
            continue

    # 若文件已含 Time_elapsed 就复用;否则重新计算(以首行时间为零点,分钟)
    if 'Time_elapsed' not in df.columns:
        start_time = df['Time'].iloc[0]
        df['Time_elapsed'] = (df['Time'] - start_time).dt.total_seconds() / 60

    # Apply Savitzky-Golay smoothing to key columns
    if len(df) >= 61:  # Only apply if we have enough data points
        df['Temperature_smoothed'] = apply_savgol_filter(df['Temperature'])
        df['Concentration_smoothed'] = apply_savgol_filter(df['Concentration'])
        df['Turbidity_smoothed'] = apply_savgol_filter(df['Turbidity'])
        df['Supersaturation_ln_smoothed'] = apply_savgol_filter(df['Supersaturation_ln'])
    else:
        # If not enough data, use original values
        df['Temperature_smoothed'] = df['Temperature']
        df['Concentration_smoothed'] = df['Concentration']
        df['Turbidity_smoothed'] = df['Turbidity']
        df['Supersaturation_ln_smoothed'] = df['Supersaturation_ln']

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
        temp_line, = ax[i, 0].plot(df['Time_elapsed'], df['Temperature_smoothed'], color=c[0], label='Temperature')
        ax[i, 0].set_title(f'({letters[i * 3 + 0]}) {rate} °C/min - Temperature & Turbidity')
        ax[i, 0].set_xlabel('Time (min)')
        ax[i, 0].set_ylabel('Temperature (°C)', color=c[0])
        ax[i, 0].tick_params(axis='y', labelcolor=c[0])
        ax[i, 0].grid(True, linestyle='--')

        # 画副轴：浊度
        ax2 = ax[i, 0].twinx()
        turb_line, = ax2.plot(df['Time_elapsed'], df['Turbidity_smoothed'], color=c[1], label='Turbidity')
        ax2.set_ylabel('Turbidity', color=c[1])
        ax2.tick_params(axis='y', labelcolor=c[1])

        # 合并 legend
        lines = [temp_line, turb_line]
        labels = [line.get_label() for line in lines]
        ax[i, 0].legend(lines, labels, loc='center left')  # 或 loc='best'


        # 2. Plot Concentration, Solubility, and MSZW
        ax[i, 1].plot(df['Time_elapsed'], df['Concentration_smoothed'], label='Concentration', color=c[1])
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
            below_threshold = df[df['Temperature_smoothed'] <= threshold_temp]
            if not below_threshold.empty:
                t_nucleation = below_threshold['Time_elapsed'].iloc[0]
            else:
                t_nucleation = df['Time_elapsed'].max()
        else:
            t_nucleation = df['Time_elapsed'].max()

        # 创建布尔掩码：过饱和 + 时间小于成核时间
        conc_mask = df['Concentration_smoothed'] > df['Solubility (g/L)']
        time_mask = df['Time_elapsed'] <= t_nucleation
        metastable_mask = conc_mask & time_mask  # 是一个和 df 一样长的 Boolean Series

        # 填充图中灰色区域表示 MSZW
        ax[i, 1].fill_between(df['Time_elapsed'],
                            df['Concentration_smoothed'],
                            df['Solubility (g/L)'],
                            where=metastable_mask,
                            color='gray',
                            alpha=0.5,
                            label='MSZW')

        
        ax[i, 1].set_title(f'({letters[i * 3 + 1]}) {rate} °C/min - Concentration vs Solubility')
        ax[i, 1].set_xlabel('Time (min)')
        ax[i, 1].set_ylabel('g/L')
        ax[i, 1].grid(True, linestyle='--')
        ax[i, 1].legend()

        # 3. Plot Supersaturation
        ax[i, 2].plot(df['Time_elapsed'], df['Supersaturation_ln_smoothed'], color=c[3])
        ax[i, 2].set_title(f'({letters[i * 3 + 2]}) {rate} °C/min - Supersaturation')
        ax[i, 2].set_xlabel('Time (min)')
        ax[i, 2].set_ylabel('Supersaturation')
        ax[i, 2].grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig('verification.tif',dpi=300)
    plt.savefig('vertification.pdf')
    plt.show()

else:
    print(f"No CSV files found in the directory: {folder_path}")