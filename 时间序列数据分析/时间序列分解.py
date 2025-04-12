"""

seasonal_decompose(
参数：
    x,                  # 输入的时间序列数据（必须为 Pandas Series 或 DataFrame）
    model='additive',   # 模型类型：'additive':加法模型（默认）或 'multiplicative'：乘法模型
    period=None,        # 季节性周期长度（必须指定，例如周期为7天）
    filt=None,          # 自定义移动平均滤波器系数（默认使用简单移动平均）
    two_sided=True,     # 是否使用双边移动平均（对称窗口，默认True）
    extrapolate_trend=0 # 是否外推趋势（填充边缘缺失值）
)

返回值：
# 输出属性：
result.observed    # 原始序列（val）
result.trend       # 趋势项（T_t）
result.seasonal    # 季节性项（S_t）
result.resid       # 残差项（R_t）

注意：移动平均时间序列分解对复杂季节性（多重周期）不适用，多重周期下应改为STL分解
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

time_date = pd.DataFrame()
# 假设 date 是一个包含多列时间序列的 DataFrame，索引为日期
for column_name in time_date:
    val = time_date[column_name].tail(30) # 取最近30天数据
    result = seasonal_decompose(val, model='additive', period=7) #    # 采用加法模型，周期7天，假设是日数据且周季节性
    result.plot()
    plt.suptitle(f'"{column_name}" 时间序列分解（周期=7天）', y=1.02)  # 标题位置微调
    plt.tight_layout()  # 防止子图重叠
    plt.savefig(f"./decompose_plots/{column_name}_分解.png",
                bbox_inches='tight', dpi=300)
    plt.close()  # 关闭当前图像，避免内存泄漏


"""
结果分析：
若残差项出现明显规律性波动（非随机），可能表明模型未完全捕捉趋势或季节性，需尝试调整 period 或切换为乘法模型（model='multiplicative'）
如果图像中趋势或季节性部分为空值（NaN），检查数据长度是否不足 2×period，或首尾因移动平均窗口导致的缺失（可设置 extrapolate_trend=1 填充边缘

"""
