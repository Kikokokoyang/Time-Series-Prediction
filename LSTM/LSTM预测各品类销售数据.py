"""
python环境：3.9
LSTM 预测模型
单层LSTM单元数：6
全连接层神经元数：7
迭代次数：400？？？？不是50吗哈哈哈
损失函数：均方误差
优化器：Adam梯度下降算法
样本批量：1
"""
"""
提高模型的性能：超参数的调整,
1.增加LSTM单元数和层数
2.调整全连接层的神经元数量
3.更改时间步长（look_back）
4.调整学习率和优化器：可以尝试调整Adam优化器的学习率（optimizer=Adam(learning_rate=0.001)），或者使用不同的优化器（如SGD, RMSprop等）
5.增加训练轮数（epochs）
6.调整批量大小（batch_size）
history = model.fit(trainX, trainY, epochs=400, batch_size=32, verbose=2)
7.进行特征选择
8.数据增强（）
9.模型调参工具：网格搜索或贝叶斯优化等工具自动调整模型的超参数，以找到最佳的参数组合
"""

"""
判断模型的过拟合和欠拟合，并且操作防止过拟合
1.如果模型在训练集上的误差（如损失值）显著低于验证集上的误差，则可能发生了过拟合。过拟合的模型在训练集上表现很好，但在验证集或测试集上表现较差。
2.解决过拟合：L2正则化+添加Dropout层
例如：
from tensorflow.keras.layers import Dropout
model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
model.add(Dropout(0.2))  # 添加 Dropout 层，丢弃20%的神经元
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(20))
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 导入评估模型性能的指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# 设置 matplotlib 绘图的字体和坐标轴负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------读取文件并格式化-----------------------------------------------
# 读取 EXCEL 文件中的数据
df = pd.read_excel('增加品类名称的批发价格.xlsx','Sheet4')
print(df.head(5)) # 打印数据集的前 5 行，用于查看数据集的头部信息检查是否正确

# 将销售日期列转化为日期时间格式
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 读取所需要用到的列值
# data = df[['销售日期','花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']].values
# 选取第二列到第七列的数据
selected_data = df.iloc[:, 1:7]
# 打印选取的列名
selected_columns = df.columns[1:7]
print("选取的列名:", selected_columns)
# 打印选取的数据的前几行
print("选取的数据:")
print(selected_data.head())  # 默认显示前5行，您可以传递一个数字参数来显示更多或更少的行

# 选择对应的蔬菜品类数据
predicted_category = '辣椒类'
df_selected = df[[predicted_category]]

# ----------------------------------------------------数据预处理：归一化------------------------------------------
# 数据归一化，将批发价格的列的值缩放（MinMaxScaler表示最大最小缩放）在0-1之间
# feature_range是一个参数，用于指定缩放范围
# fit_transform是MinMaxScaler的一个方法，它首先计算数据的最小值和最大值（fit），然后使用这些值来转换（transform）数据

# reshape（-1，1）将一维数组重塑为二维数组（1，2，3，4，5，6，7，8，9）变成竖着的（1，2，3，4，5，6，7，8，9）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['辣椒类'].values.reshape(-1, 1))
"""
保留意见
"""


# ----------------------------------------------开始建模，利用LSTM训练和预测-----------------------------------------------------
# 划分训练集的大小，0.8
train_size = int(len(scaled_data) * 0.8)  # 返回scaled_data数组的长度
train = scaled_data[0:train_size, :]  # 数组中选择从索引0开始到 train_size 结束（不包括 train_size）的所有行，以及所有列
teat = scaled_data [train_size :,:] # 后20%是测试集


# 定义函数，转换数据格式为符合 LSTM 输入要求
# LSTM模型的输入必须是三维数组，其形状为 [样本数量, 时间步长, 特征数量]
# dataset：这是原始数据的二维数组，其中每一行代表一个时间步，每一列代表一个特征
# look_back：这是每个输入样本的时间步长，即LSTM模型将考虑的过去时间步数
# dataX 和 dataY：分别存储输入特征和目标值。
# dataX 存储从当前时间步到 look_back 时间步的特征，而 dataY 存储下一个时间步的目标值
"""
时间步长的选择（look_back）
如果想同时捕捉多个特征，可以引入注意力机制、多尺度时间步长、滑动窗口逐步移动时间步长、
数据分解为趋势、季节和随机成分，对这些进行建模，或者将他们作为特征输入LSTM中、
周期性数据增强：如果数据集中的周期性模式不够明显，可以通过数据增强技术来增加更多的周期性样本。例如，可以通过复制和旋转数据点来创建额外的周期性样本
混合模型：结合傅里叶变换或者自回归模型
"""
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    # 在每个时间步 i 处，我们都能取到 look_back 个连续的时间步作为输入
    # 这里的 - look_back - 1 确保了在每个时间步 i 处，我们可以安全地取到从 i 开始的 look_back 个连续时间步，并且不会超出数组的边界
    # 确保了对于每个样本，我们都有一个后续的目标值可以用于训练
    for i in range(len(dataset) - look_back - 1):
        # 切片：从 dataset 中提取从时间步 i 开始的 look_back 个连续时间步的数据。
        # 这里的索引 0 表示我们只关心 dataset 中的第一个特征（列）。a 将是一个包含 look_back 个数据点的数组。
        a = dataset[i:(i + look_back), 0]
        # 将提取的数据 a 添加到 dataX 列表中
        dataX.append(a)
        # 对于每个时间步 i，使用 dataset[i + look_back, 0] 获取 look_back 时间步之后的下一个时间步的数据点，作为目标值，并将其添加到 dataY 列表中
        dataY.append(dataset[i + look_back, 0])
        # dataX 变成了一个三维数组，其形状为 [样本数量, 时间步长, 特征数量]。这是 LSTM 网络所期望的输入数据格式。
        # dataY 变成了一个二维数组，其形状为 [样本数量, 特征数量]。这代表了每个输入样本的目标值。
    return np.array(dataX), np.array(dataY)

look_back = 30  # 时间步长选择30
# 这个函数通常将原始数据序列转换为监督学习的训练数据，其中每个训练样本包含过去的 30 个时间点的数据，而标签则是第 31 个时间点的数据。
trainX, trainY = create_dataset(train, look_back)
# 构建预测样本：这行代码去除训练数据的最后30个时间点
pre_X = train[-30:]
# 将最后 30 个时间点的数据扁平化，形成一个一维数组
pre_X = [item for sublist in pre_X for item in sublist]
pre_X = np.array([[pre_X]])
# 将训练数据集 trainX 重塑为三维数组。格式化时间序列预测模型的输入格式，样本数+时间序列长度（这里只有一个时间序列，所以是1）+特征数
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# 构建 LSTM 模型
# 首先构建一个顺序模型：创建模型实例，Sequential是一个用于线性堆叠多个网络层的模型。
model = Sequential()
# 添加一个LSTM层，LSTM（6）创建了一个LSTM层，其中6表示LSTM层中的神经元数量
# input_shape=(1, look_back) 指定了输入数据的形状。这里 1 表示每个时间步长的特征数，look_back 是时间步长（在这个例子中是 30）
model.add(LSTM(6, input_shape=(1, look_back)))
# 创建全连接层，Dense（7），7表示输出神经元的数量，（这个数字一般与目标变量的数量相匹配）
model.add(Dense(7))
# 编译模型；loss指定模型的损失函数，optimizer指定优化器
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型，epochs指定训练的次数，batch_size指定每次训练的样本数，1表示逐个样本训练，verbose指定输出详细程度
# verbose 参数的值决定了训练过程中输出多少信息到控制台或日志文件
history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

# 预测训练数据
trainPredict = model.predict(trainX)
# 将预测结果从标准化值转化为原始的值
trainPredict = scaler.inverse_transform(trainPredict)
# 将训练标签从标准化值转化为原始值
trainY = scaler.inverse_transform([trainY])

# 预测未来数据
pre_Y = model.predict(pre_X)
pre_Y = scaler.inverse_transform(pre_Y)

# 计算 R 方、MSE、RMSE 和 MAE
# 计算均方误差
mse = mean_squared_error(trainY[0], trainPredict[:, 0])
# 计算均方根误差
rmse = sqrt(mse)
# 计算平均绝对误差
mae = mean_absolute_error(trainY[0], trainPredict[:, 0])
# 计算R^2
r2 = r2_score(trainY[0], trainPredict[:, 0])

print(f'R2: {r2}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
# 咋保证的刚好预测七个呢？
print(f" 预测未来七天{predicted_category}的销量：{pre_Y[0, :]}")

# 创建一个 figure 和 axes 对象
fig, ax = plt.subplots(figsize=(12, 6))
# 绘制实际值和预测值，trainY[0] 和 trainPredict[:, 0] 分别是实际值和预测值的数据
ax.plot(trainY[0], label=f'{predicted_category} 实际进价')
ax.plot(trainPredict[:, 0], label=f'{predicted_category} 预测进价')
# 设置图表的标签和标题
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend() # 显示图例
# 设置图表标题
ax.set_title(f'{predicted_category} LSTM 预测结果图')
# 添加网格线
ax.grid()

# 保存图像
fig.savefig(f"{predicted_category}LSTM预测.png")
fig.show()

# 创建一个新的 figure 和 axes 对象，用于绘制损失曲线
fig, ax = plt.subplots(figsize=(12, 6))
loss_history = history.history['loss']
ax.plot(loss_history, label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title(f'{predicted_category}Loss Over Training Epochs')
ax.grid()

fig.savefig(f"{predicted_category}Loss.png")
fig.show()
