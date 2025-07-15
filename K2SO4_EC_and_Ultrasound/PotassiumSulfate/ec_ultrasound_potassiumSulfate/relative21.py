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
plt.rcParams['font.size'] = 12                   # 全局字体大小
plt.rcParams['axes.titlesize'] = 12              # 图标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
c = sns.color_palette("deep")

dataec = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\0.1.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
dataultrasonic = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\C36916-DATALOG-2025-01-04-18-48-39.xlsx')
relativeEc = pd.read_csv(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\ecRelativeerror.csv')
relativeUltrasonic = pd.read_csv(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\ultrasoundfitrelative_error.csv')

temUltrasonic = dataultrasonic['温度 - 传感器 1 (50627)']
temEc = dataec.iloc[:, 2]
relativeEc2 = relativeEc.iloc[0:790, 1]
relativeUltrasonic2 = relativeUltrasonic.iloc[0:58,1]

print(temUltrasonic.shape)
print(temEc.shape)
print(relativeEc2.shape)
print(relativeUltrasonic2.shape)

fig = plt.figure(figsize = (12,5))

ax1 = fig.add_subplot(121)
ax1.plot(temEc, relativeEc2, label = 'Relative Errors of EC model in 21.5 g/L', color = c[0])
ax1.set_xlabel('Temperature (℃)')
ax1.set_ylabel('Relative Error (%)')
ax1.legend()
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.text(0.1, 0.95, '(A)', transform=ax1.transAxes, fontsize=14, va='top', ha='right')

ax2 = fig.add_subplot(122)
ax2.plot(temUltrasonic, relativeUltrasonic2, label = 'Relative Errors of Ultrasonic Model in 21.5 g/L', color = c[0])
ax2.set_xlabel('Temperature (℃)')
ax2.set_ylabel('Relative Error (%)')
ax2.legend()
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.text(0.1, 0.85, '(B)', transform=ax2.transAxes, fontsize=14, va='top', ha='right')

plt.savefig(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\relativeErrors.pdf', dpi = 300, bbox_inches='tight')
plt.show()