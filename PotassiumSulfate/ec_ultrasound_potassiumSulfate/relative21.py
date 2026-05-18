import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.stats as stats

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14                   # 全局字体大小
plt.rcParams['axes.titlesize'] = 14              # 图标题字体大小
plt.rcParams['axes.labelsize'] = 14              # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14             # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14             # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 14             # 图例字体大小
c = sns.color_palette("deep")

dataec = pd.read_excel(r'/home/kemove/WorkpaceP2/junjie/K2SO4Ultrasound/PotassiumSulfate/ec_ultrasound_potassiumSulfate/0.1_0104/0.1.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
dataultrasonic = pd.read_excel(r'/home/kemove/WorkpaceP2/junjie/K2SO4Ultrasound/PotassiumSulfate/ec_ultrasound_potassiumSulfate/0.1_0104/C36916-DATALOG-2025-01-04-18-48-39.xlsx')
relativeEc = pd.read_csv(r'ecRelativeerror.csv')
relativeUltrasonic = pd.read_csv(r'ultrasoundfitrelative_error.csv')

temUltrasonic = dataultrasonic['温度 - 传感器 1 (50627)']
temEc = dataec.iloc[:, 2]
relativeEc2 = relativeEc.iloc[0:790, 1]
relativeUltrasonic2 = relativeUltrasonic.iloc[6:58,1]

print(temUltrasonic.shape)
print(temEc.shape)
print(relativeEc2.shape)
print(relativeUltrasonic2.shape)
print(f"EC温度范围: {temEc.min():.2f} - {temEc.max():.2f}")
print(f"超声温度范围: {temUltrasonic.min():.2f} - {temUltrasonic.max():.2f}")

# 对数据进行插值对齐到统一的温度网格
from scipy.interpolate import interp1d

# 确定统一的温度范围
temp_min = max(temEc.min(), temUltrasonic.min())
temp_max = min(temEc.max(), temUltrasonic.max())

# 创建统一的温度网格
temp_unified = np.linspace(temp_min, temp_max, 100)

# 对EC数据进行插值
f_ec = interp1d(temEc, relativeEc2, kind='linear', fill_value='extrapolate')
relativeEc_interp = f_ec(temp_unified)

# 对超声数据进行插值
f_ultrasonic = interp1d(temUltrasonic, relativeUltrasonic2, kind='linear', fill_value='extrapolate')
relativeUltrasonic_interp = f_ultrasonic(temp_unified)

# 在两边留出5%的空白
temp_range = temp_max - temp_min
margin = temp_range * 0.05

fig = plt.figure(figsize = (8,6))

ax = fig.add_subplot(111)
ax.plot(temp_unified, relativeEc_interp, label = 'EC Model', color = c[0], linewidth=1.5)
ax.plot(temp_unified, relativeUltrasonic_interp, label = 'Ultrasonic Model', color = c[1], linewidth=1.5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Relative Error (%)')
ax.set_xlim(temp_min - margin, temp_max + margin)
ax.legend(loc='best')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.savefig(r'relativeErrors.pdf', dpi = 300, bbox_inches='tight')
plt.show()