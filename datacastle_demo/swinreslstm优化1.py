import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Dense, LSTM, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def convert2oneHot(labels, num_classes=10):
    """将标签转换为 one-hot 编码"""
    labels = labels.astype(int)
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"标签超出了合法的类别范围 (0 到 {num_classes - 1})")
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, window_size, num_heads, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.norm1(attn_output + inputs)
        ffn_output = self.ffn(attn_output)
        return self.norm2(ffn_output + attn_output)

    def get_config(self):
        config = super(SwinTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'window_size': self.window_size,
            'num_heads': self.num_heads
        })
        return config


def ResNeXtBlock(inputs, filters, cardinality):
    group_channels = filters // cardinality
    group_list = []
    for _ in range(cardinality):
        x = Conv1D(group_channels, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        group_list.append(x)
    x = layers.Concatenate()(group_list)
    x = Conv1D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def build_model(input_shape=(6000, 1), num_classes=10, depth=2, cardinality=8, embed_dim=32, window_size=4, num_heads=2,
                dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    x = SwinTransformerBlock(embed_dim=embed_dim, window_size=window_size, num_heads=num_heads)(inputs)
    x = ResNeXtBlock(x, filters=64, cardinality=cardinality)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Ensure float32 for numerical stability
    model = Model(inputs, outputs)
    return model


class DataGenerator(Sequence):
    def __init__(self, df, batch_size=32, num_classes=10, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        # 提取特征和标签
        batch_x = batch_df.iloc[:, 1:-1].astype(float).values
        batch_x = batch_x.reshape(-1, 6000, 1)
        batch_y = convert2oneHot(batch_df.iloc[:, -1].values, self.num_classes)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data(path, train=True, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    if train:
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df.iloc[:, -1])
        return train_df, val_df
    else:
        return df


def main():
    # 加载并划分数据
    train_df, val_df = load_data("train.csv", train=True)

    # 创建数据生成器
    batch_size = 32
    train_gen = DataGenerator(train_df, batch_size=batch_size, num_classes=10, shuffle=True)
    val_gen = DataGenerator(val_df, batch_size=batch_size, num_classes=10, shuffle=False)

    # 构建模型
    model = build_model(input_shape=(6000, 1), num_classes=10)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 定义回调
    checkpoint_callback = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # 训练模型
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )

    # 保存最终模型
    model.save('final_model.h5')

    # 加载测试数据并预测
    test_df = load_data('test_data.csv', train=False)

    # 假设测试数据的特征与训练数据相同
    test_gen = DataGenerator(test_df, batch_size=batch_size, num_classes=10, shuffle=False)
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'SampleID': test_df.index,
        'PredictedClass': predicted_classes
    })
    predictions_df.to_csv('predictions.csv', index=False)

    print("预测完成，结果已保存为 'predictions.csv'")


if __name__ == "__main__":
    main()
