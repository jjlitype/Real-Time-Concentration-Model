import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.optimize import fsolve

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12                   # 全局字体大小
plt.rcParams['axes.titlesize'] = 12              # 图标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
c = sns.color_palette("deep")

data01 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0102\0.1_0102.xlsx', names = ['time', 'ec','temperature'],  usecols=[0,1,2])
data02 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\ec1230-origin.xlsx', names = ['time', 'ec','temperature'],  usecols=[0,1,2])

T_data01 = data01.iloc[:, 2].values
kappa_data01 = data01.iloc[:, 1].values
T_data02 = data02.iloc[:, 2].values
kappa_data02 = data02.iloc[:, 1].values
time_in_min01 = [(t - data01.iloc[:, 0][0]).total_seconds() / 60 for t in data01.iloc[:, 0]]
time_in_min02 = [(t - data02.iloc[:, 0][0]).total_seconds() / 60 for t in data02.iloc[:, 0]]

P1, P2, P3, P4, n = 9.32536839e-03 , 2.42789146e-01, -3.32128577e-01, -1.06149975e+02, 1.18097731e+00

def ec_concentration(C, kappa, T, P1, P2, P3, P4, n):
    return (P1 * T + P2) * C**n * np.exp(P3 * C / (T - P4)) - kappa

c_initial_guess = 20

c_solutions01 = []
c_solutions02 = []

for kappa, T in zip(kappa_data01, T_data01):
    c_solution01 = fsolve(ec_concentration, c_initial_guess, args=(kappa, T, P1, P2, P3, P4, n))
    c_solutions01.append(c_solution01[0]) 

for kappa, T in zip(kappa_data02, T_data02):
    c_solution02= fsolve(ec_concentration, c_initial_guess, args=(kappa, T, P1, P2, P3, P4, n))
    c_solutions02.append(c_solution02[0]) 

data03 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0102\C36916-DATALOG-2025-01-03-10-16-09.xlsx')
data04 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\C36916-DATALOG-2024-12-31-11-04-32origin.xlsx')

def ultrasonic_concentration(T, v, A1, A2, A3, A4, A5, A6):
    return A1 + A2*T + A3*v + A4*T**2 + A5*v**2 + A6*T*v
A1, A2, A3, A4, A5, A6 =  3.28922186e+03, -8.14108529e+00, -5.37345189e+00,  3.31028220e-02, 2.17057787e-03,  2.10709059e-03

T03 = data03['温度 - 传感器 1 (50627)']
T04 = data04['温度 - 传感器 1 (50627)']
v03 = data03['声速 - 传感器 1 (50627)']
v04 = data04['声速 - 传感器 1 (50627)']
time03 = data03['日期和时间 ']
time04 = data04['日期和时间 ']
time_in_min03 = [(t - time03[0]).total_seconds() / 60 for t in time03]
time_in_min04 = [(t - time04[0]).total_seconds() / 60 for t in time04]

c_solutions03 = ultrasonic_concentration(T03, v03, A1, A2, A3, A4, A5, A6)
c_solutions04 = ultrasonic_concentration(T04, v04, A1, A2, A3, A4, A5, A6)

# fig = plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# plt.plot(time_in_min01,c_solutions01, color=c[0],label = 'Concentration by EC')
# plt.plot(time_in_min03,c_solutions03, color=c[2],label = 'Concentration by Ultrasonic')
# plt.xlabel('Time (min)')
# plt.ylabel('Concentration (g/L)')
# ax2 = plt.twinx()
# plt.plot(time_in_min01, T_data01, color=c[1],label = 'Temperature')
# plt.ylabel('Temperature (℃)')
# lines_1, labels_1 = ax.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
fig = plt.figure(figsize=(6, 5))
ax3 = plt.subplot(111)
plt.plot(time_in_min02,c_solutions02,color=c[0],label = 'Concentration by EC model')
plt.plot(time_in_min04,c_solutions04,color=c[1],label = 'Concentration by Ultrasonic model')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (g/L)')
ax3.grid(True, linestyle='--', linewidth=0.5)
ax4 = plt.twinx()
plt.plot(time_in_min02, T_data02, color=c[2],label = 'Temperature')
plt.ylabel('Temperature (℃)')
lines_1, labels_1 = ax3.get_legend_handles_labels()
lines_2, labels_2 = ax4.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2,loc='lower left')
plt.tight_layout()
plt.savefig(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\crystalProcess.pdf', dpi=300, bbox_inches='tight')
plt.show()

# c_solution01_pd = pd.DataFrame(c_solution01, columns=['c'])
# c_solution01_pd.to_csv(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\c_solution.csv', index=False)