import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 模拟数据加载
data = np.array([
    [0.2286, 0.1292, 0.072, 0.1592, 0.1335, 0.0733, 0.1159, 0.094, 0.0522, 0.1345, 0.009, 0.126, 0.3619, 0.069, 0.1828],
    [0.209, 0.0947, 0.1393, 0.1387, 0.2558, 0.09, 0.0771, 0.0882, 0.0393, 0.143, 0.0126, 0.167, 0.245, 0.0508, 0.1328],
    [0.0442, 0.088, 0.1147, 0.0563, 0.3347, 0.115, 0.1453, 0.0429, 0.1818, 0.0378, 0.0092, 0.2251, 0.1516, 0.0858, 0.067],
    [0.2603, 0.1715, 0.0702, 0.2711, 0.1491, 0.133, 0.0968, 0.1911, 0.2545, 0.0871, 0.006, 0.1793, 0.1002, 0.0789, 0.0909],
    [0.369, 0.2222, 0.0562, 0.5157, 0.1872, 0.1614, 0.1425, 0.1506, 0.131, 0.05, 0.0078, 0.0348, 0.0451, 0.0707, 0.088],
    [0.0359, 0.1149, 0.123, 0.546, 0.1977, 0.1248, 0.0624, 0.0832, 0.164, 0.1002, 0.0059, 0.1503, 0.1837, 0.1295, 0.07],
    [0.1759, 0.2347, 0.1829, 0.1811, 0.2922, 0.0655, 0.0774, 0.2273, 0.2056, 0.0925, 0.0078, 0.1852, 0.3501, 0.168, 0.2668],
    [0.0724, 0.1909, 0.134, 0.2409, 0.2842, 0.045, 0.0824, 0.1064, 0.1909, 0.1586, 0.0116, 0.1698, 0.3644, 0.2718, 0.2494],
    [0.2634, 0.2258, 0.1165, 0.1154, 0.1074, 0.0657, 0.061, 0.2623, 0.2588, 0.1155, 0.005, 0.0978, 0.1511, 0.2273, 0.322]
])


# 模拟标签加载
labels = np.array(['无故障', '无故障', '无故障', '齿根裂纹', '齿根裂纹', '齿根裂纹', '断齿', '断齿', '断齿'])


# 将标签转换为数值
label_mapping = {'无故障': 0, '齿根裂纹': 1, '断齿': 2}
labels_numeric = np.array([label_mapping[label] for label in labels])

# 数据归一化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels_numeric, test_size=0.33, random_state=42)

# 构建RNN模型
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], 1), activation='relu'))
model.add(Dense(3, activation='softmax'))  # 假设有3个类别

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=100, batch_size=16, verbose=2)

# 评估模型
loss, accuracy = model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test, verbose=0)
print(f"Accuracy: {accuracy}")

# 预测
y_pred = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
y_pred_classes = np.argmax(y_pred, axis=1)
print("Test results:")
for i, pred in enumerate(y_pred_classes):
    print(f"Data {i+1}: Predicted label: {pred}, Actual label: {y_test[i]}")