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

data01 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\C36916-DATALOG-2025-01-04-18-48-39.xlsx',usecols=[0,1,2,3,4])
data02 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.2_0104\C36916-DATALOG-2025-01-04-21-50-01.xlsx',usecols=[0,1,2,3,4])
data03 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.3_0103\C36916-DATALOG-2025-01-03-14-17-10.xlsx',usecols=[0,1,2,3,4])
data045 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.45_0104\C36916-DATALOG-2025-01-05-13-26-46.xlsx',usecols=[0,1,2,3,4])
data06 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.6_0103\C36916-DATALOG-2025-01-03-17-51-04.xlsx',usecols=[0,1,2,3,4])
data80 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\80_0105\C36916-DATALOG-2025-01-05-21-50-56.xlsx',usecols=[0,1,2,3,4])
data120 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\120_0106\C36916-DATALOG-2025-01-06-13-32-38.xlsx',usecols=[0,1,2,3,4])
data085 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\C36916-DATALOG-2024-12-31-11-04-32.xlsx',usecols=[0,1,2,3,4])

data01.insert(0,'concentration',21.5)
data02.insert(0,'concentration',36.9)
data03.insert(0,'concentration',50.8)
data045.insert(0,'concentration',65.0)
data80.insert(0,'concentration',85.5)
data06.insert(0,'concentration',104.5)
data120.insert(0,'concentration',121.7)
data085.insert(0,'concentration',156.0)

dataAll = pd.concat([data01, data02, data03, data045, data80, data06, data120, data085], ignore_index=True)
print(dataAll.columns)

def ultrasonic_model( T, v, A1, A2, A3, A4, A5, A6):
    return A1 + A2*T + A3*v + A4*T**2 + A5*v**2 + A6*T*v
C_data = dataAll.iloc[:, 0].values
T_data = dataAll.iloc[:, 4].values
v_data = dataAll.iloc[:, 5].values

dataAll.iloc[:,[0,5,4]].to_csv('ultrasoundAll.csv', index=False, header= True)

# print(C_data.shape, T_data.shape, v_data.shape)
x_data = np.vstack((T_data, v_data)).T
# print(x_data.shape)
initial_guess = np.array([1 ,1, 1, 1, 1, 1])

popt, pcov = curve_fit(
    lambda x, A1,A2,A3,A4,A5,A6: ultrasonic_model(x[:,0], x[:,1], A1,A2,A3,A4,A5,A6),
    x_data, C_data, p0=initial_guess
)

perr = np.sqrt(np.diag(pcov))
print("Fitted Parameters:", popt)
print("Parameter Errors:", perr)

confidence_interval = 1.96 * perr

# 计算拟合值
yfit = ultrasonic_model(x_data[:,0], x_data[:,1], *popt)

# 计算残差
residuals = C_data - yfit

# 计算均方误差 
mse = np.mean(residuals**2)
print(f"均方误差 (MSE): {mse}")

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"均方根误差 (RMSE): {rmse}")

# 计算 R^2 值
ss_res = np.sum(residuals**2)
ss_tot = np.sum((C_data - np.mean(C_data))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R^2 值: {r2}")

fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(3,2,1)
plt.scatter(C_data, yfit, color=c[0],label = 'Fitted Data')
plt.plot([min(C_data), max(C_data)], [min(C_data), max(C_data)], color=c[1], linestyle='--',label = 'Ideal Line')  # 理想的拟合线
plt.xlabel(r'Observed Concentration (g/L)')
plt.ylabel(r'Fitted Concentration (g/L)')
# plt.title('Observed vs Fitted Plot')
plt.text(0.05,0.95, '(A)', transform=ax1.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
plt.legend(loc = 'lower right')


# 残差
ax2 = fig.add_subplot(3,2,2)
plt.scatter(C_data, residuals, color=c[0], label='Residuals', s=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # 添加水平线 y=0
plt.xlabel('Concentration (g/L)')
plt.ylabel('Residuals (g/L)')
# plt.legend()
plt.text(0.05,0.95, '(B)', transform=ax2.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')


# 相对误差
ax3 = fig.add_subplot(3,2,3)
relative_error = np.abs( residuals / C_data) * 100
plt.scatter(C_data, relative_error, color=c[0], label='Relative error', s=10)
plt.xlabel('Concentration (g/L)')
plt.ylabel('Relative Error (%)')
# plt.legend()
plt.axhline(5, color='black', linestyle='--', linewidth=1)  
plt.text(
    x=160,  # 文字的 x 坐标（调整为适合的位置）
    y=5,  # 文字的 y 坐标（与水平线相同）
    s="Threshold = 5%",  # 文本内容
    fontsize=14,
    verticalalignment="bottom",  # 对齐方式
    horizontalalignment="right",  # 对齐方式
)
plt.text(0.05,0.95, '(C)', transform=ax3.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')

ax4 = fig.add_subplot(3,2,4)
plt.hist(residuals, bins=30, color=c[0], edgecolor='black', alpha=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # 添加零值参考线
plt.xlabel('Residuals (g/L)')
plt.ylabel('Frequency')
# plt.title('Residuals Histogram')
plt.text(0.05,0.95, '(D)', transform=ax4.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')

ax5 = plt.subplot(3,2,5)
stats.probplot(residuals, dist="uniform", plot=plt)
plt.text(0.05,0.95, '(E)', transform=ax5.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
line = ax5.get_lines()[0]  # 获取第一条线（正态拟合线）
line.set_color(c[0])     # 设置颜色为蓝色
line.set_label('Residuals Samples')
line2 = ax5.get_lines()[1] 
line2.set_color(c[1])
line2.set_linestyle('--')
line2.set_label('Ideal Line of Uniform Distribution')
plt.title('') 
plt.legend(loc = 'lower right')

# line.set_linewidth(2)      # 设置线宽

ax6 = plt.subplot(3,2,6)
stats.probplot(residuals, dist="norm", plot=plt)
plt.text(0.05,0.95, '(F)', transform=ax6.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
line = ax6.get_lines()[0]  # 获取第一条线（正态拟合线）
line.set_color(c[0])     # 设置颜色为蓝色
line.set_label('Residuals Samples')
line2 = ax6.get_lines()[1]
line2.set_color(c[1])
line2.set_linestyle('--')
line2.set_label('Ideal Line ofNormal Distribution')
plt.title('') 
plt.legend(loc = 'lower right')


plt.tight_layout()
plt.savefig(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\ultrasonicfit.pdf', dpi=300, bbox_inches='tight')

RelativeError = pd.DataFrame(relative_error)
RelativeError.to_csv(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\ultrasoundfitrelative_error.csv')

plt.show()