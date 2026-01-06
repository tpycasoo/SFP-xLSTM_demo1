
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Dense, LSTM, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os


import numpy as np
import pandas as pd
import math
# 在 convert2oneHot 函数中确保标签是整数类型
def convert2oneHot(label, num_classes=10):
    """将标签转换为 one-hot 编码"""
    label = int(label)  # 强制转换为整数
    if label < 0 or label >= num_classes:
        raise ValueError(f"标签 {label} 超出了合法的类别范围 (0 到 {num_classes-1})")
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot



# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, window_size, num_heads):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads

        # MultiHeadAttention 自注意力机制
        self.attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        # Feed-forward 网络
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])

    def call(self, inputs):
        # 自注意力层
        attn_output = self.attention(inputs, inputs)
        attn_output = self.norm1(attn_output + inputs)  # 残差连接

        # 前馈网络
        ffn_output = self.ffn(attn_output)
        return self.norm2(ffn_output + attn_output)  # 残差连接

    def get_config(self):
        config = super(SwinTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'window_size': self.window_size,
            'num_heads': self.num_heads
        })
        return config

# 简单的 ResNeXt Block
def ResNeXtBlock(inputs, filters, cardinality):
    group_list = []
    group_channels = filters // cardinality

    for _ in range(cardinality):
        x = Conv1D(group_channels, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        group_list.append(x)

    # Concatenate groups
    x = layers.Concatenate()(group_list)
    x = Conv1D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# 定义 convert2oneHot 函数
def convert2oneHot(label, num_classes):
    """将标签转换为 one-hot 编码"""
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

# 定义模型
def build_model(input_shape=(6000, 1), num_classes=10, depth=2, cardinality=8, embed_dim=32, window_size=4, num_heads=2):
    inputs = Input(shape=input_shape)

    # 1. Swin Transformer
    x = SwinTransformerBlock(embed_dim=embed_dim, window_size=window_size, num_heads=num_heads)(inputs)

    # 2. ResNeXt
    x = ResNeXtBlock(x, filters=64, cardinality=cardinality)

    # 3. LSTM
    x = LSTM(64, return_sequences=False)(x)

    # 4. Dropout 层
    x = Dropout(0.3)(x)

    # 5. 输出层
    outputs = Dense(num_classes, activation='softmax')(x)

    # 构建模型
    model = Model(inputs, outputs)
    return model



# 数据生成器
def xs_gen(path, batch_size=5, train=True, Lens=640):
    # 读取数据
    img_list = pd.read_csv(path)

    # 如果是训练数据，取前 Lens 个样本，否则取后面的样本
    if train:
        img_list = np.array(img_list)[:Lens]
    else:
        img_list = np.array(img_list)[Lens:]

    # 计算每个批次的步数
    steps = math.ceil(len(img_list) / batch_size)

    while True:
        for i in range(steps):
            # 获取当前批次数据
            batch_list = img_list[i * batch_size: (i + 1) * batch_size]
            np.random.shuffle(batch_list)  # 打乱顺序

            # 特征数据 (去掉第一列和最后一列，提取6000个特征)
            batch_x = np.array([row[1:-1].astype(float) for row in batch_list])  # 特征列
            batch_x = batch_x.reshape(-1, 6000, 1)  # 调整形状为 (batch_size, 6000, 1)

            # 在 xs_gen 中确保从 batch_list 中提取的标签是整数
            batch_y = np.array([convert2oneHot(int(label), num_classes=10) for label in batch_list[:, -1]])  # 标签列

            yield batch_x, batch_y


# 使用训练数据生成器
train_iter = xs_gen(path="train.csv", train=True)

# 编译并训练模型
model = build_model(input_shape=(6000, 1), num_classes=10)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_callback = ModelCheckpoint(
    filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

model.fit(
    train_iter,
    steps_per_epoch=32,
    epochs=10,
    validation_data=train_iter,
    validation_steps=32,
    callbacks=[checkpoint_callback]
)
model.save('final_model.h5')

# 1. 加载训练好的模型
model = tf.keras.models.load_model('final_model.h5')

# 2. 读取测试数据
test_data = pd.read_csv('test_data.csv')


# 如果没有额外的预处理步骤，假设特征已经在 CSV 中是直接可用的：
X_test = test_data.values  # 如果 `test_data.csv` 只是特征，没有标签

# 4. 使用模型进行预测
predictions = model.predict(X_test)

# 5. 假设是分类任务，预测结果是类别的概率分布，可以使用 argmax 来获取类别标签
predicted_classes = np.argmax(predictions, axis=1)

# 如果你希望将预测结果与对应的样本编号一起输出，可以创建一个 DataFrame
predictions_df = pd.DataFrame({
    'SampleID': test_data.index,  # 假设每一行是一个样本
    'PredictedClass': predicted_classes
})

# 6. 保存预测结果到 CSV 文件
predictions_df.to_csv('predictions.csv', index=False)

print("预测完成，结果已保存为 'predictions.csv'")
