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

data01 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.1_0104\0.1.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data02 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.2_0104\0.2.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data03 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.3_0103\0.3_0103.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data045 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.45_0104\0.45.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data06 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.6_0103\0.6.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data80 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\80_0105\80.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data120 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\120_0106\120.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])
data085 = pd.read_excel(r'PotassiumSulfate\ec_ultrasound_potassiumSulfate\0.85_1230\ec1230.xlsx', names = ['time', 'ec','temperature'],header= None,usecols=[0,1,2])

# print(data01.shape, data02.shape, data03.shape, data045.shape, data80.shape, data06.shape, data120.shape, data085.shape)

data01.insert(3,'concentration',21.5)
data02.insert(3,'concentration',36.9)
data03.insert(3,'concentration',50.8)
data045.insert(3,'concentration',65.0)
data80.insert(3,'concentration',85.5)
data06.insert(3,'concentration',104.5)
data120.insert(3,'concentration',121.7)
data085.insert(3,'concentration',156.0)
# print(data01.shape, data02.shape, data03.shape, data045.shape, data06.shape, data80.shape, data120.shape, data085.shape)
dataAll = pd.concat([data01, data02, data03, data045, data80, data06, data120, data085], ignore_index=True)

C_data = dataAll.iloc[:, 3].values
T_data = dataAll.iloc[:, 2].values
kappa_data = dataAll.iloc[:, 1].values
x_data = np.vstack((T_data, kappa_data)).T
initial_guess = [ 9.56099175e-03,  2.40513916e-01, -3.82437460e-01, -1.32143500e+02, 1.17735553e+00]

def fitting_model(x, P1, P2, P3, P4, n):
    T, kappa = x[:, 0], x[:, 1]
    # Guess for C during fitting
    C_guess = 20
    for _ in range(1000):  # Iterate to refine C
        C_guess = (P1 * T + P2) * C_guess**n * np.exp(P3 * C_guess / (T - P4)) - kappa
    return C_guess

# Fit the model
popt, pcov = curve_fit(
    lambda x, P1, P2, P3, P4, n: fitting_model(x, P1, P2, P3, P4, n),
    x_data,
    C_data,
    p0=initial_guess,
)
