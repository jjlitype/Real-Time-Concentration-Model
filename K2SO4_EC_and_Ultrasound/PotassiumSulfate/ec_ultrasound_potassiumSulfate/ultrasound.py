import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12                   # 全局字体大小
plt.rcParams['axes.titlesize'] = 12              # 图标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
c = sns.color_palette("deep")

data01 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\C36916-DATALOG-2025-01-04-18-48-39.xlsx')
data02 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.2_0104\C36916-DATALOG-2025-01-04-21-50-01.xlsx')
data03 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.3_0103\C36916-DATALOG-2025-01-03-14-17-10.xlsx')
data045 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.45_0104\C36916-DATALOG-2025-01-05-13-26-46.xlsx')
data06 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.6_0103\C36916-DATALOG-2025-01-03-17-51-04.xlsx')
data80 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\80_0105\C36916-DATALOG-2025-01-05-21-50-56.xlsx')
data120 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\120_0106\C36916-DATALOG-2025-01-06-13-32-38.xlsx')
data085 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\C36916-DATALOG-2024-12-31-11-04-32.xlsx')

dataAll = pd.concat([data01, data02, data03, data045, data06, data80, data120, data085])


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data01['温度 - 传感器 1 (50627)'], data01['声速 - 传感器 1 (50627)'], c=c[0], label='21.5 g/L')
ax.plot(data02['温度 - 传感器 1 (50627)'], data02['声速 - 传感器 1 (50627)'], c=c[1], label='36.9 g/L')
ax.plot(data03['温度 - 传感器 1 (50627)'], data03['声速 - 传感器 1 (50627)'], c=c[2], label='50.8 g/L')
ax.plot(data045['温度 - 传感器 1 (50627)'], data045['声速 - 传感器 1 (50627)'], c=c[3], label='65.0 g/L')
ax.plot(data80['温度 - 传感器 1 (50627)'], data80['声速 - 传感器 1 (50627)'], c=c[4], label='85.5 g/L')
ax.plot(data06['温度 - 传感器 1 (50627)'], data06['声速 - 传感器 1 (50627)'], c=c[5], label='104.5 g/L')
ax.plot(data120['温度 - 传感器 1 (50627)'], data120['声速 - 传感器 1 (50627)'], c=c[6], label='121.7 g/L')
ax.plot(data085['温度 - 传感器 1 (50627)'], data085['声速 - 传感器 1 (50627)'], c=c[7], label='156.0 g/L')
ax.set_xlabel('Temperature (℃)')
ax.set_ylabel('Sound Velocity (m/s)')
plt.ylim(1465,1645)
ax.legend(ncol = 2)
ax.grid(True, linestyle='--', linewidth=0.5)


plt.savefig(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\ultrasonic.pdf',format='pdf', dpi=300, bbox_inches='tight')
plt.show()