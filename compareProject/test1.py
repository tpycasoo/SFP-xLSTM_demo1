import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# TensorFlow 导入
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dropout, Dense, LSTM, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# 禁用 GPU，强制使用 CPU
tf.config.set_visible_devices([], 'GPU')

# 简单的 Swin Transformer Block
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

# 定义模型
def build_model(input_shape=(6000, 1), num_classes=10, depth=3, cardinality=8, embed_dim=32, window_size=8, num_heads=4):
    inputs = Input(shape=input_shape)

    # 1. Swin Transformer
    x = SwinTransformerBlock(embed_dim=embed_dim, window_size=window_size, num_heads=num_heads)(inputs)
    # 输出形状：(batch_size, 6000, embed_dim)

    # 2. ResNeXt
    x = ResNeXtBlock(x, filters=128, cardinality=cardinality)
    # 输出形状：(batch_size, 6000, filters)

    # 3. LSTM 输入需要的是 (batch_size, timesteps, features)
    # 不需要 Reshape，只需直接传递给 LSTM
    x = LSTM(64, return_sequences=False)(x)
    # 输出形状：(batch_size, 128)

    # 4. Dropout 层
    x = Dropout(0.3)(x)
    # 输出形状：(batch_size, 128)

    # 5. 输出层
    outputs = Dense(num_classes, activation='softmax')(x)
    # 输出形状：(batch_size, num_classes)

    # 构建模型
    model = Model(inputs, outputs)
    return model

# 编译并查看模型结构
input_shape = (6000, 1)
model = build_model(input_shape=input_shape, num_classes=10)
model.summary()


# 数据生成器
def xs_gen(path, batch_size=20, train=True, Lens=640):
    img_list = pd.read_csv(path)
    if train:
        img_list = np.array(img_list)[:Lens]
    else:
        img_list = np.array(img_list)[Lens:]
    steps = math.ceil(len(img_list) / batch_size)

    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: (i + 1) * batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([row[1:-1].astype(float) for row in batch_list])  # 特征列
            batch_x = batch_x.reshape(-1, 6000, 1)  # 调整形状
            batch_y = np.array([convert2oneHot(label, 10) for label in batch_list[:, -1]])
            yield batch_x, batch_y

def convert2oneHot(index, num_classes):
    hot = np.zeros((num_classes,))
    hot[int(index)] = 1
    return hot


# 模型架构（假设模型已经定义）
model = build_model(input_shape=(6000, 1), num_classes=10, depth=3, cardinality=8, embed_dim=32, window_size=8, num_heads=4)

# 使用plot_model绘制模型结构
#plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# 训练模型
if __name__ == "__main__":
    model = build_model(input_shape=(6000, 1), num_classes=10, depth=3, cardinality=8, embed_dim=32, window_size=8, num_heads=4)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    checkpoint_callback = ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    train_iter = xs_gen(path="D:/桌面/zhoucheng/轴承数据/train.csv", train=True)
    val_iter = xs_gen(path="D:/桌面/zhoucheng/轴承数据/train.csv", train=False)

    model.fit(
        train_iter,
        steps_per_epoch=32,
        epochs=10,
        validation_data=val_iter,
        validation_steps=32,
        callbacks=[checkpoint_callback]
    )

    model.save('final_model.h5')
