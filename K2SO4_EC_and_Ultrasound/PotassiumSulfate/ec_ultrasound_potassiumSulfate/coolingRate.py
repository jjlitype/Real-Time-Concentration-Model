import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
from datetime import datetime

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12                   # 全局字体大小
plt.rcParams['axes.titlesize'] = 12              # 图标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
c = sns.color_palette("deep")

data01 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\0.1.xlsx')
data02 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.2_0104\0.2.xlsx')
data03 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.3_0103\0.3_0103.xlsx')
data045 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.45_0104\0.45.xlsx')
data06 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.6_0103\0.6.xlsx')
data085 = pd.read_excel(r'C:\Users\64838\Documents\Matplotlib\PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\ec1230.xlsx')
time_in_min01 = [(t - data01.iloc[:, 0][0]).total_seconds() / 60 for t in data01.iloc[:, 0]]

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(time_in_min01, data01.iloc[:, 2], c=c[0], label='21.5 g/L')
# ax.plot(data02.iloc[:, 2], data02.iloc[:, 1], c=c[1], label='36.9 g/L')
# ax.plot(data03.iloc[:, 2], data03.iloc[:, 1], c=c[2], label='50.8 g/L')
# ax.plot(data045.iloc[:, 2], data045.iloc[:, 1], c=c[3], label='65.0 g/L')
# ax.plot(data06.iloc[:, 2], data06.iloc[:, 1], c=c[4], label='104.5 g/L')
# ax.plot(data085.iloc[:, 2], data085.iloc[:, 1], c=c[5], label='156.0 g/L')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Conductivity (mS/m)')
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5)

plt.savefig('coolingRate.tif', dpi=300, bbox_inches='tight')
plt.show()
