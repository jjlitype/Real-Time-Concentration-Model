import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns

# --- 样式设置 (保持不变) ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

c = sns.color_palette("deep")

# --- 数据计算 (保持不变) ---
x = range(0,100) # x 的范围是 0 到 99
a0 = 4.18e-1
a1 = 1.138e-2
a2 = -1.688e-5
# 计算 y 值列表，对应 x 从 0 到 99
y = [(a0 + a1 * i + a2 * i**2)*174.26 for i in x]

# --- (可选) 其他计算，根据需要保留或注释掉 ---
# yg = [(a0 + a1 * i + a2 * i**2) * 174.26  for i in x]
# y30 = [(a0 + a1 * 30 + a2 * 30**2) * 174.26]
# print(y30)

# --- 将 x 和 y 存储到 CSV ---
# 创建一个 Pandas DataFrame
# 列名分别为 'Temperature_C' 和 'Calculated_Y_Value'
df_output = pd.DataFrame({
    'Temperature_C': list(x),  # 将 range 对象转为列表作为第一列
    'Calculated_Y_Value': y   # y 值列表作为第二列
})

# 定义要保存的 CSV 文件名
csv_filename = 'temperature_y_values.csv'

# 将 DataFrame 保存为 CSV 文件
# index=False 表示不将 DataFrame 的索引写入 CSV 文件
# encoding='utf-8-sig' 通常能更好地处理包含非英文字符的情况，尤其是在Excel中打开时
df_output.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"数据已成功保存到文件: {csv_filename}")

# --- (可选) 绘图代码，根据需要保留或注释掉 ---
# fig = plt.figure(figsize=(6, 5))
# axs = fig.add_subplot(111)
# axs.plot(x,yg,label = '$K_2SO_4$', color = c[0])
# axs.set_xlabel('Temperature (°C)')
# axs.set_ylabel('Solubility (g/L)')
# axs.grid(True, linestyle='--', linewidth=0.5)
# axs.legend()
# plt.savefig('K2SO4Soubility3.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()