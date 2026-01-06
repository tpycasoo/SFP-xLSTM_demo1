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
        raise ValueError(f"标签 {label} 超出了合法的类别范围 (0 到 {num_classes - 1})")
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Swin Transformer Block
class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, window_size, num_heads, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads

        # MultiHeadAttention 自注意力机制
        self.attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
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


# ResNeXt Block
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


# Importance1D Layer
class Importance1D(layers.Layer):
    def __init__(self, dim, kernel_size=3, strides=1, padding='same', groups=1, **kwargs):
        super(Importance1D, self).__init__(**kwargs)
        self.conv = layers.Conv1D(filters=dim, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)
        self.act = layers.Activation('gelu')
        self.ln = layers.LayerNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.ln(x)
        return x


# SparseFocalModulation1D Layer
class SparseFocalModulation1D(layers.Layer):
    def __init__(self, dim, focal_level=3, focal_x=[3, 1, 1, 1], focal_factor=2, proj_drop=0.0, **kwargs):
        super(SparseFocalModulation1D, self).__init__(**kwargs)
        self.dim = dim
        self.focal_level = focal_level
        self.focal_x = focal_x[0]
        self.focal_factor = focal_factor

        self.f = layers.Dense(2 * dim + (focal_level + 1), use_bias=True)
        self.h = layers.Conv1D(filters=dim, kernel_size=1, padding='same', use_bias=True)
        self.act = layers.Activation('gelu')
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

        self.focal_layers = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_x
            self.focal_layers.append(
                Importance1D(dim, kernel_size=kernel_size, padding='same')
            )

    def avg_pool_gate(self, ctx, gate):
        # Global average pooling along the temporal dimension
        ctx = tf.reduce_mean(ctx, axis=1, keepdims=True)
        return ctx * gate

    def call(self, x):
        C = x.shape[-1]
        x_fea = self.f(x)
        q, ctx_fea, gates = tf.split(x_fea, [C, C, self.focal_level + 1], axis=-1)
        ctx = ctx_fea

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all += ctx * tf.expand_dims(gates[..., l], axis=-1)

        ctx_global = self.avg_pool_gate(ctx, gates[..., self.focal_level:])
        ctx_all += ctx_global

        ctx_all = self.h(ctx_all)
        x_out = q * ctx_all

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


# ResFBlock1D Layer
class ResFBlock1D(layers.Layer):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 focal_level=3, focal_x=[3, 1, 1, 1], focal_factor=2, **kwargs):
        super(ResFBlock1D, self).__init__(**kwargs)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_x = focal_x
        self.focal_level = focal_level

        self.norm1 = layers.LayerNormalization()
        self.modulation = SparseFocalModulation1D(dim=dim,
                                                  focal_level=focal_level,
                                                  focal_x=focal_x,
                                                  focal_factor=focal_factor,
                                                  proj_drop=drop)
        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else lambda x: x
        self.norm2 = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.modulation(x)
        x = shortcut + self.drop_path(x)
        focal_x = x
        focal_x = focal_x + self.drop_path(self.mlp(self.norm2(focal_x)))
        return focal_x


# 定义模型
def build_model(input_shape=(6000, 1), num_classes=10, depth=2, cardinality=8,
                embed_dim=32, window_size=4, num_heads=2, focal_level=3):
    inputs = Input(shape=input_shape)

    # 1. Swin Transformer
    x = SwinTransformerBlock(embed_dim=embed_dim, window_size=window_size, num_heads=num_heads)(inputs)

    # 2. ResFBlock1D
    x = ResFBlock1D(dim=embed_dim, focal_level=focal_level)(x)

    # 3. ResNeXt
    x = ResNeXtBlock(x, filters=64, cardinality=cardinality)

    # 4. LSTM
    x = LSTM(64, return_sequences=False)(x)

    # 5. Dropout 层
    x = Dropout(0.3)(x)

    # 6. 输出层
    outputs = Dense(num_classes, activation='softmax')(x)

    # 构建模型
    model = Model(inputs, outputs)
    return model


# 数据生成器
def xs_gen(path, batch_size=5, train=True, Lens=640):
    # 读取数据
    img_list = pd.read_csv(path).values

    # 如果是训练数据，取前 Lens 个样本，否则取后面的样本
    if train:
        img_list = img_list[:Lens]
    else:
        img_list = img_list[Lens:]

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
model = tf.keras.models.load_model('final_model.h5', compile=False)

# 2. 读取测试数据
test_data = pd.read_csv('test_data.csv')

# 如果没有额外的预处理步骤，假设特征已经在 CSV 中是直接可用的：
X_test = test_data.values[:, 1:-1].astype(float)  # 假设第一列是样本ID，最后一列是标签
X_test = X_test.reshape(-1, 6000, 1)  # 调整形状为 (num_samples, 6000, 1)

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
