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

dataAll = pd.read_csv(r'ecAll.csv')

# \kappa(C,T) = (P_1T +P_2)C^n exp(\frac{P_3C}{T-P_4})

def conductivity_model( T, C, P1, P2, P3, P4, n):
    return (P1 * T + P2) * C**n * np.exp((P3 * C) / (T - P4))

C_data = dataAll.iloc[:, 0].values
T_data = dataAll.iloc[:, 2].values
kappa_data = dataAll.iloc[:, 1].values
# print(C_data.shape, T_data.shape, kappa_data.shape)
x_data = np.vstack((T_data, C_data)).T
# print(x_data.shape)
initial_guess = np.array([1 ,1, 1, 1, 1])

popt, pcov = curve_fit(
    lambda x, P1, P2, P3, P4, n: conductivity_model(x[:,0], x[:,1], P1, P2, P3, P4, n),
    x_data, kappa_data, p0=initial_guess
)

perr = np.sqrt(np.diag(pcov))
print("Fitted Parameters:", popt)
print("Parameter Errors:", perr)

confidence_interval = 1.96 * perr

# 计算拟合值
yfit = conductivity_model(x_data[:,0], x_data[:,1], *popt)

# 计算残差
residuals = kappa_data - yfit

# 计算均方误差 
mse = np.mean(residuals**2)
print(f"均方误差 (MSE): {mse}")

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"均方根误差 (RMSE): {rmse}")

# 计算 R^2 值
ss_res = np.sum(residuals**2)
ss_tot = np.sum((kappa_data - np.mean(kappa_data))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R^2 值: {r2}")

fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(3,2,1)
plt.scatter(kappa_data, yfit, color=c[0],label = 'Fitted data', s=10)
plt.plot([min(kappa_data), max(kappa_data)], [min(kappa_data), max(kappa_data)], color=c[1], linestyle='--', label = 'Ideal line')  # 理想的拟合线
plt.xlabel(r'Observed $\kappa$ (ms/cm)')
plt.ylabel(r'Fitted $\kappa$ (ms/cm)')
# plt.title('Observed vs Fitted Plot')
plt.text(0.05,0.95, '(A)', transform=ax1.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')
plt.legend(loc='lower right')


# 残差
ax2 = fig.add_subplot(3,2,2)
plt.scatter(C_data, residuals, color=c[0], label='Residuals', s=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # 添加水平线 y=0
plt.xlabel('Concentration (g/L)')
plt.ylabel('Residuals (ms/cm)')
# plt.legend()
plt.text(0.05,0.95, '(B)', transform=ax2.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')


# 相对误差
ax3 = fig.add_subplot(3,2,3)
relative_error = np.abs( residuals / kappa_data) * 100
plt.scatter(C_data, relative_error, color=c[0], label='Relative error', s=10)
plt.axhline(5, color='black', linestyle='--', linewidth=1)  
plt.text(
    x=160,  # 文字的 x 坐标（调整为适合的位置）
    y=5,  # 文字的 y 坐标（与水平线相同）
    s="Threshold = 5%",  # 文本内容
    fontsize=14,
    verticalalignment="bottom",  # 对齐方式
    horizontalalignment="right",  # 对齐方式
)
plt.xlabel('Concentration (g/L)')
plt.ylabel('Relative Error (%)')
# plt.legend()
plt.text(0.05,0.95, '(C)', transform=ax3.transAxes, fontsize=14, fontweight='normal', va='top', ha='left')

ax4 = fig.add_subplot(3,2,4)
plt.hist(residuals, bins=30, color=c[0], edgecolor='black', alpha=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # 添加零值参考线
plt.xlabel('Residuals (ms/cm)')
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
line2.set_label('Ideal line of Uniform Distribution')
line2.set_linestyle('--')
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
line2.set_label('Ideal line of Normal Distribution')
line2.set_linestyle('--')
plt.title('') 
plt.legend(loc = 'lower right')

RelativeError = pd.DataFrame(relative_error)
RelativeError.to_csv(r'ecRelativeerror.csv')

plt.tight_layout()
plt.savefig(r'ecfit.pdf', dpi=300, bbox_inches='tight')
plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# ax.plot(data01.iloc[:, 2], data01.iloc[:, 1], c=c[0], label='21.5 g/L')
# ax.plot(data02.iloc[:, 2], data02.iloc[:, 1], c=c[1], label='36.9 g/L')
# ax.plot(data03.iloc[:, 2], data03.iloc[:, 1], c=c[2], label='50.8 g/L')
# ax.plot(data045.iloc[:, 2], data045.iloc[:, 1], c=c[3], label='65.0 g/L')
# ax.plot(data80.iloc[:, 2], data80.iloc[:, 1], c=c[4], label='85.5 g/L')
# ax.plot(data06.iloc[:, 2], data06.iloc[:, 1], c=c[5], label='104.5 g/L')
# ax.plot(data120.iloc[:, 2], data120.iloc[:, 1], c=c[6], label='121.7 g/L')
# ax.plot(data085.iloc[:, 2], data085.iloc[:, 1], c=c[7], label='156.0 g/L')
# ax.set_xlabel('Temperature (℃)')
# ax.set_ylabel('Conductivity (mS/cm)')
# ax.legend(ncol=2)
# ax.grid(True, linestyle='--', linewidth=0.5)

# plt.savefig('ec.pdf',format='pdf', dpi=300, bbox_inches='tight')
# plt.show()
