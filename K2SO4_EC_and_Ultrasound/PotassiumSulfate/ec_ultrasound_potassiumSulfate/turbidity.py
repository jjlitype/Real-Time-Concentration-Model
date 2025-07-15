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
color = sns.color_palette("deep")

turbidity = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\turbidity\156.xlsx', header = 0, names = ['time', 'turbidity'])
tem = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\turbidity\C36916-DATALOG-2025-01-13-18-46-46.xlsx')
tem1 = tem.iloc[:, 3].values
time1 = tem.iloc[:, 0].values

time1_in_min = [(t - tem.iloc[:, 0][0]).total_seconds() / 60 for t in tem.iloc[:, 0]]
time_in_min01 = [(t - turbidity.iloc[:, 0][0]).total_seconds() / 60 for t in turbidity.iloc[:, 0]]


fig= plt.figure(figsize=(12, 5))
ax = plt.subplot(121)
line1 = ax.plot(time1_in_min, tem1, label = 'Temperature',c=color[0])
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (℃)')
ax.grid(linestyle='--', linewidth=0.5)
ydot = 27
xdot = 31.5
plt.scatter(xdot, ydot, s=100, c=color[2], marker='x')
plt.annotate(
    f'({xdot}, {ydot})',
    xy=(xdot, ydot),
    xytext=(xdot + 5, ydot - 2),  # 偏移位置
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=14,
    color='black'
)
plt.text(0.05,0.9, '(A)', transform=ax.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
ax2 = ax.twinx()
line2 = ax2.plot(time_in_min01, turbidity['turbidity'], label = 'Turbidity',c=color[1])
ax2.axhline(200, color='black', linestyle='--', linewidth=1)
ax2.axvline(31.5, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('Turbidity (NTU)')
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc = 'upper right')

axs = plt.subplot(122)
x = range(0,100)
a0 = 4.18e-1
a1 = 1.138e-2
a2 = -1.688e-5
yg = [(a0 + a1 * i + a2 * i**2) * 174.26  for i in x]
y = [(a0 + a1 * 20 + a2 * 20**2) * 174.26]
x_pd = pd.DataFrame(x)
y_pd = pd.DataFrame(yg)
xy_pd = pd.concat([x_pd, y_pd], axis=1)
print(xy_pd)
xy_pd.to_csv(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\turbidity\solubility.csv', index = False)
axs.plot(x,yg,label = 'K$_2$SO$_4$ Solubility', color = color[0])
axs.set_xlabel('Temperature (°C)')
axs.set_ylabel('Solubility (g/L)')
axs.grid(True, linestyle='--', linewidth=0.5)
# axs.legend()
plt.text(0.05,0.9, '(B)', transform=axs.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
xdot2 = 45
ydot2 = 156
plt.scatter([xdot2,27], [ydot2,156], s=100, c=color[1], marker='x')
plt.scatter([27], [156], s=100, c=color[1], marker='x')
plt.annotate(
    '', xy=[xdot2,ydot2], xytext=[27,156],
    arrowprops=dict(arrowstyle="<|-|>", color=color[0], lw=1)
)
plt.annotate(
    f'({xdot2}, {ydot2})',
    xy=(xdot2, ydot2),
    xytext=(xdot2 + 10, ydot2 - 10),  # 偏移位置
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=14,
    color='black'
)
plt.annotate(
    f'(27,156)',
    xy=(27,156),
    xytext=(27 - 20, 156 + 10),  # 偏移位置
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=14,
    color='black'
)
plt.text(32,170,'MSZ',fontsize=14, fontweight='normal', va='top', ha='left')
plt.tight_layout()
plt.savefig(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\turbidity.pdf', dpi=300, bbox_inches='tight')
plt.show()