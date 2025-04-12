"""
原理是假设检验
H0假设为：存在单位根
要拒绝原假设，才能判定序列平稳
拒绝原假设输出p值要求：p小于0.05（对应95%的置信度）

函数关键参数：
x：一维的数据序列。
maxlag：最大滞后数目。
regresults：True 完整的回归结果将返回。False，默认。

一部分返回值：
adf：Test statistic，T检验，假设检验值。
pvalue：假设检验结果。
usedlag：使用的滞后阶数。

"""
from statsmodels.tsa.stattools import adfuller as adf
print(" 平稳性检验：")
for column_name in times_data:
    for column_name in times_data:
        print(column_name)
        result = adf(times_data[column_name])  # 进行ADF检验
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])