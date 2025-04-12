"""
ACF函数要求：
输入数据结构：一维时间序列（索引日期+序列）
数据要求：数值型，不含缺失值

函数关键参数：
pmdarima.acf(
    x, 输入的一维时间序列数据
    nlags=None,算自相关的最大滞后阶数。默认 nlags=None，自动选择 min(10 * log10(n), n - 1)，其中 n 是数据长度
    fft=False,是否使用快速傅里叶变换 (FFT) 加速计算
    alpha=None,
    bartlett_confint=True,
    missing='none'
)

返回值：从滞后0到滞后n阶的自相关系数
"""
import pandas as pd
from pmdarima import acf
import matplotlib.pyplot as plt

times_date = pd.DataFrame()

print(" 自相关函数 (ACF)：")
for colname in times_date:
    acf_values = acf(times_date[colname].tail(60))  # 计算自相关函数（取最近60天的数据）
    plt.figure()
    plt.plot(acf_values)  # 绘制自相关函数图
    plt.title(f'ACF of {colname} (Last 60 Days)')
    plt.savefig(f"{colname}ACF(天).png")  # 保存图像
    plt.close()
