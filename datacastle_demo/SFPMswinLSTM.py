import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, Dropout, Dense, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional, Generator

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class ScalerManager:
    """标准化器管理类"""

    def __init__(self):
        self.scaler = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        """拟合标准化器"""
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用标准化转换"""
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        return self.scaler.transform(data)


def data_loader(file_path: str,
                batch_size: int = 32,
                scaler: Optional[ScalerManager] = None,
                is_train: bool = True,
                num_classes: int = 10) -> tf.data.Dataset:
    """
    高效数据加载器

    Args:
        file_path: 数据文件路径
        batch_size: 批次大小
        scaler: 标准化器实例
        is_train: 是否训练模式
        num_classes: 类别数量

    Returns:
        TF Dataset对象
    """
    df = pd.read_csv(file_path)

    # 提取特征和标签
    if is_train:
        features = df.iloc[:, 1:-1].values.astype(np.float32)
        labels = df.iloc[:, -1].values.astype(np.int32)
    else:
        features = df.iloc[:, 1:].values.astype(np.float32)
        labels = np.zeros(len(df))  # 测试集伪标签

    # 标准化处理
    if is_train and scaler is not None:
        scaler.fit(features)
    if scaler and scaler.is_fitted:
        features = scaler.transform(features)

    # 转换标签为one-hot
    labels = to_categorical(labels, num_classes=num_classes)

    # 创建TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if is_train:
        dataset = dataset.shuffle(buffer_size=len(features))

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class GlobalToken(layers.Layer):
    """全局令牌生成层"""

    def __init__(self, num_global_tokens: int = 2, embed_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_global_tokens = num_global_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape: tf.TensorShape) -> None:
        self.global_tokens = self.add_weight(
            shape=(self.num_global_tokens, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='global_tokens'
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        global_tokens = tf.tile(
            tf.expand_dims(self.global_tokens, axis=0),
            [batch_size, 1, 1]
        )
        return tf.concat([global_tokens, inputs], axis=1)

    def get_config(self) -> dict:
        return super().get_config().update({
            'num_global_tokens': self.num_global_tokens,
            'embed_dim': self.embed_dim
        })


class WindowAttention(layers.Layer):
    """滑动窗口注意力机制"""

    def __init__(self, window_size: int = 100, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=64)

    def window_partition(self, x: tf.Tensor) -> tf.Tensor:
        """将输入划分为窗口"""
        B, L, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        num_windows = (L + pad_len) // self.window_size
        x = tf.reshape(x, [B, num_windows, self.window_size, D])
        return x, pad_len

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x, pad_len = self.window_partition(inputs)
        B, N, W, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * N, W, D])
        attn_output = self.attention(x, x)
        attn_output = tf.reshape(attn_output, [B, N, W, D])
        return attn_output, pad_len


class SwinTransformerBlock(layers.Layer):
    """改进的Swin Transformer块"""

    def __init__(self, embed_dim: int = 64, window_size: int = 100,
                 num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.window_attention = WindowAttention(window_size, num_heads)
        self.global_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim)
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # 窗口注意力
        x = self.norm1(inputs)
        window_output, pad_len = self.window_attention(x)

        # 残差连接
        x = inputs + window_output

        # 全局注意力
        global_output = self.global_attention(x, x)
        x = x + global_output

        # MLP
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        return inputs + mlp_output


def build_optimized_model(input_shape: Tuple[int, int] = (6000, 1),
                          num_classes: int = 10,
                          embed_dim: int = 64,
                          num_blocks: int = 4,
                          window_size: int = 100,
                          num_heads: int = 8) -> Model:
    """构建优化后的模型架构"""
    inputs = Input(shape=input_shape, dtype=tf.float32)

    # 输入投影
    x = layers.Dense(embed_dim)(inputs)

    # 添加全局令牌
    x = GlobalToken(embed_dim=embed_dim)(x)

    # 堆叠多个Swin Transformer块
    for _ in range(num_blocks):
        x = SwinTransformerBlock(embed_dim=embed_dim,
                                 window_size=window_size,
                                 num_heads=num_heads)(x)

    # 特征提取
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)

    # 输出层（显式使用float32保证数值稳定性）
    outputs = layers.Dense(num_classes, activation='softmax', dtype=tf.float32)(x)

    model = Model(inputs, outputs)
    return model


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """带热身的余弦衰减学习率调度"""

    def __init__(self, initial_lr: float = 1e-3, warmup_steps: int = 1000,
                 decay_steps: int = 10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.pi = tf.constant(math.pi)

    def __call__(self, step: int) -> tf.Tensor:
        warmup_lr = self.initial_lr * (tf.cast(step, tf.float32) / self.warmup_steps)
        decay_factor = 0.5 * (1 + tf.cos(self.pi * (step - self.warmup_steps) / self.decay_steps))
        decay_lr = self.initial_lr * decay_factor
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)


def main():
    # 配置参数
    config = {
        'input_shape': (6000, 1),
        'num_classes': 10,
        'embed_dim': 64,
        'window_size': 100,
        'num_heads': 8,
        'num_blocks': 4,
        'batch_size': 128,
        'epochs': 50,
        'warmup_epochs': 2
    }

    # 初始化标准化器
    scaler = ScalerManager()

    # 加载数据
    train_df = pd.read_csv('train.csv')
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # 创建数据管道
    train_data = data_loader('train.csv', config['batch_size'], scaler, is_train=True)
    val_data = data_loader('val.csv', config['batch_size'], scaler, is_train=False)

    # 构建模型
    model = build_optimized_model(
        input_shape=config['input_shape'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_blocks=config['num_blocks'],
        window_size=config['window_size'],
        num_heads=config['num_heads']
    )

    # 配置优化器
    total_steps = config['epochs'] * len(train_df) // config['batch_size']
    warmup_steps = config['warmup_epochs'] * len(train_df) // config['batch_size']
    lr_schedule = CosineDecayWithWarmup(
        initial_lr=1e-3,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps
    )
    optimizer = Adam(lr_schedule)

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 配置回调
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    # 训练模型
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config['epochs'],
        callbacks=callbacks
    )

    # 保存最终模型
    model.save('final_model.h5')


if __name__ == "__main__":
    main()