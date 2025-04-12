"""
模型选择依据
"""

import numpy as np
import pandas as pd

# 定义R/S重标极差函数（每个子序列）
'''
time_subseries:一维数值型数组（无缺失值）
'''
def rs_analysis(time_subseries):
    N = len(time_subseries)
    T = np.arange(1, N + 1)
    mean_ts = np.mean(time_subseries) # 计算每个子区间的均值
    Z = np.cumsum(time_subseries - mean_ts) # 计算每个子区间的累积求和
    R = np.max(Z) - np.min(Z) # 极差
    S = np.std(time_subseries) # 标准差
    return R / S

# 定义计算Hurst指数的函数（拟合幂律关系）
def hurst_exponent(time_series):
    N = len(time_series)
    max_k = int(np.log2(N)) # 定义子区间个数最多为 max_k,用于保证每个子区间数值大于2
    R_S = [] # 初始化R/S列表
    for k in range(2, max_k + 1): # k = 2，3，……
        M = int(N / k) # 子区间的长度
        R_S_k = []
        for i in range(k): # 分割为k个子区间
            R_S_k.append(rs_analysis(time_series[i * M: (i + 1) * M])) # 计算每个子区间的R/S
        R_S.append(np.mean(R_S_k))  # 保存当前k对应的平均R/S
    T = np.arange(2, max_k + 1)
    H = np.polyfit(np.log(T), np.log(R_S), 1)[0]  # 拟合斜率即Hurst指数
    return H

# 函数使用示例
"""
假设分析销量（数据结构：日期+销量）
"""
data = pd.read_excel("sales_data.xlsx", parse_dates=["日期"])  # 读取并解析日期列
time_series = data["销量（千克）"].values  # 提取销量列为一维数组
H = hurst_exponent(time_series)


# ------------------------------------------
"""
可以直接调用hurst.compute_Hc函数计算
参数：
data：时间序列
kind:指定计算方法（price：输入数据为价格序列，函数内部会转化为收益率；和random_walk：数据输入为平稳增量序列）
simplified：
    False: 使用完整的 R/S 分析方法，生成多个时间窗口并计算每个窗口的 R/S 值。
    True: 使用简化算法，可能跳过部分窗口或减少计算步骤（牺牲精度换取速度）

返回值：
H:hurst指数
c：拟合常数部分
R_squared： 回归模型的拟合优度
"""
import hurst
H, c, R2 = hurst.compute_Hc(data, kind='price')

# 结果解读
if H > 0.6:
    print("序列具有强趋势性（建议使用趋势模型如Holt-Winters）")
elif H > 0.5:
    print("序列具有弱趋势性（建议使用ARIMA）")
elif H < 0.4:
    print("序列均值回复（建议使用均值回归策略）")
else:
    print("序列接近随机游走（无显著趋势或均值回复）")