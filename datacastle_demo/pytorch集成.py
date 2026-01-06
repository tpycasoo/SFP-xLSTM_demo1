import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import math
import os
import copy


# ----------------------------
# 模型组件定义
# ----------------------------

class Mlp(nn.Module):
    """
    前馈神经网络（MLP）
    """

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Importance1D(nn.Module):
    """
    多层次上下文提取模块，使用 Conv1D
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(Importance1D, self).__init__()
        self.ctx = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (batch, dim, seq_length)
        x = self.ctx(x)
        x = self.act(x)
        # LayerNorm expects shape (batch, seq_length, dim)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x


class SparseFocalModulation1D(nn.Module):
    """
    稀疏焦点点调制模块（1D）
    Args:
        dim (int): 输入通道数。
        focal_level (int): 焦点层级数。
        focal_x (list): 每个焦点层级的核大小。
        focal_factor (int): 焦点窗口增大的步长。
        proj_drop (float): Dropout 概率。
    """

    def __init__(self, dim, proj_drop=0., focal_level=3, focal_x=[3, 1, 1, 1],
                 focal_factor=2):
        super(SparseFocalModulation1D, self).__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.focal_x = focal_x[0]
        self.focal_factor = focal_factor

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focal_layers = nn.ModuleList()
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_x
            padding = (kernel_size - 1) // 2
            self.focal_layers.append(
                Importance1D(dim, kernel_size=kernel_size, padding=padding, groups=1, bias=True)
            )

    def avg_pool_gate(self, ctx, gate):
        # ctx shape: (batch, dim, seq_length)
        # gate shape: (batch, focal_level + 1)
        # 全局平均池化
        ctx = ctx.mean(dim=2, keepdim=True)  # (batch, dim, 1)
        gate = gate[:, self.focal_level:].unsqueeze(2)  # (batch, focal_level +1, 1)
        # 将 gate 在 dim 上求和
        gate_sum = gate.sum(dim=1, keepdim=True)  # (batch, 1, 1)
        return ctx * gate_sum  # (batch, dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_length, dim)
        x = x.permute(0, 2, 1)  # (batch, dim, seq_length)
        C = x.size(1)
        x_fea = self.f(x.permute(0, 2, 1))  # (batch, seq_length, 2*dim + focal_level +1)
        q, ctx_fea, gates = torch.split(x_fea, [C, C, self.focal_level + 1], dim=-1)
        q = q.permute(0, 2, 1)  # (batch, dim, seq_length)
        ctx = ctx_fea.permute(0, 2, 1)  # (batch, dim, seq_length)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)  # (batch, dim, seq_length)
            gate = gates[:, l].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            ctx_all += ctx * gate  # Broadcasting

        # 全局上下文
        ctx_global = self.avg_pool_gate(ctx, gates)  # (batch, dim, 1)
        ctx_all += ctx_global  # Broadcasting

        ctx_all = self.h(ctx_all)  # (batch, dim, seq_length)
        x_out = q * ctx_all  # Element-wise multiplication

        x_out = self.proj(x_out.permute(0, 2, 1))  # (batch, seq_length, dim)
        x_out = self.proj_drop(x_out)
        return x_out


class ResFBlock1D(nn.Module):
    """
    稀疏焦点点调制块（1D）
    Args:
        dim (int): 输入通道数。
        mlp_ratio (float): MLP 中隐藏层维度与输入维度的比率。
        drop (float): Dropout 概率。
        drop_path (float): DropPath 概率。
        focal_level (int): 焦点层级数。
        focal_x (list): 每个焦点层级的核大小。
        focal_factor (int): 焦点窗口增大的步长。
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 focal_level=3, focal_x=[3, 1, 1, 1], focal_factor=2):
        super(ResFBlock1D, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_x = focal_x
        self.focal_level = focal_level

        self.norm1 = nn.LayerNorm(dim)
        self.modulation = SparseFocalModulation1D(dim=dim,
                                                  focal_level=focal_level,
                                                  focal_x=focal_x,
                                                  focal_factor=focal_factor,
                                                  proj_drop=drop)
        self.drop_path = nn.Identity()
        if drop_path > 0.:
            self.drop_path = nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状 (batch, seq_length, dim)
        """
        shortcut = x
        x = self.norm1(x)
        x = self.modulation(x)
        x = shortcut + self.drop_path(x)
        focal_x = x
        focal_x = focal_x + self.drop_path(self.mlp(self.norm2(focal_x)))
        return focal_x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer 块
    """

    def __init__(self, embed_dim, window_size, num_heads, mlp_ratio=4., drop=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x shape: (batch, seq_length, embed_dim)
        x_norm1 = self.norm1(x)
        attn_output, _ = self.attention(x_norm1.transpose(0, 1), x_norm1.transpose(0, 1), x_norm1.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)  # (batch, seq_length, embed_dim)
        x = x + attn_output  # 残差连接

        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output  # 残差连接
        return x


class ResNeXtBlock(nn.Module):
    """
    简单的 ResNeXt 块
    """

    def __init__(self, in_channels, out_channels, cardinality=8):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        group_channels = out_channels // cardinality

        self.groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, group_channels, kernel_size=3, padding=1, groups=1, bias=False),
                nn.BatchNorm1d(group_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(cardinality)
        ])
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (batch, embed_dim, seq_length)
        group_outputs = [group(x) for group in self.groups]
        x = torch.cat(group_outputs, dim=1)  # (batch, out_channels, seq_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class CustomModel(nn.Module):
    """
    自定义模型，集成 ResFBlock 模块
    """

    def __init__(self, input_dim=6000, num_classes=10, embed_dim=32, window_size=4, num_heads=2,
                 focal_level=3, focal_x=[3, 1, 1, 1], focal_factor=2, cardinality=8, mlp_ratio=4., drop=0.3,
                 drop_path=0.):
        super(CustomModel, self).__init__()
        self.embed_dim = embed_dim

        # 初始 Conv1D 进行嵌入
        self.initial_conv = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm1d(embed_dim)
        self.initial_relu = nn.ReLU(inplace=True)

        # Swin Transformer 块
        self.swin_transformer = SwinTransformerBlock(embed_dim=embed_dim, window_size=window_size, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

        # ResFBlock1D
        self.res_fblock = ResFBlock1D(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                                      focal_level=focal_level, focal_x=focal_x, focal_factor=focal_factor)

        # ResNeXt 块
        self.resnext = ResNeXtBlock(in_channels=embed_dim, out_channels=64, cardinality=cardinality)

        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=False)

        # Dropout
        self.dropout = nn.Dropout(drop)

        # 输出层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, 6000, 1)
        x = x.permute(0, 2, 1)  # (batch, 1, 6000)
        x = self.initial_conv(x)  # (batch, embed_dim, 6000)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = x.permute(0, 2, 1)  # (batch, 6000, embed_dim)

        # Swin Transformer
        x = self.swin_transformer(x)  # (batch, 6000, embed_dim)

        # ResFBlock1D
        x = self.res_fblock(x)  # (batch, 6000, embed_dim)

        # ResNeXt
        x = x.permute(0, 2, 1)  # (batch, embed_dim, 6000)
        x = self.resnext(x)  # (batch, 64, 6000)
        x = x.permute(0, 2, 1)  # (batch, 6000, 64)

        # LSTM
        x, _ = self.lstm(x)  # (batch, 6000, 64)
        x = x[:, -1, :]  # (batch, 64)

        # Dropout
        x = self.dropout(x)

        # 输出
        x = self.fc(x)  # (batch, num_classes)
        return x


# ----------------------------
# 数据处理
# ----------------------------

class CustomDataset(Dataset):
    def __init__(self, csv_file, train=True, lens=640, num_classes=10):
        """
        Args:
            csv_file (string): CSV 文件路径。
            train (bool): 是否为训练数据。
            lens (int): 训练数据的样本数量。
            num_classes (int): 分类类别数。
        """
        self.data = pd.read_csv(csv_file).values
        if train:
            self.data = self.data[:lens]
        else:
            self.data = self.data[lens:]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def convert2oneHot(self, label):
        """将标签转换为 one-hot 编码"""
        one_hot = np.zeros(self.num_classes, dtype=np.float32)
        one_hot[int(label)] = 1.0
        return one_hot

    def __getitem__(self, idx):
        row = self.data[idx]
        # 假设第一列是样本ID，最后一列是标签
        features = row[1:-1].astype(np.float32)
        features = features.reshape(-1, 1)  # (6000, 1)
        label = self.convert2oneHot(row[-1])
        return torch.from_numpy(features), torch.from_numpy(label)


def get_dataloaders(train_csv, test_csv, batch_size=32, lens=640, num_classes=10):
    train_dataset = CustomDataset(csv_file=train_csv, train=True, lens=lens, num_classes=num_classes)
    val_dataset = CustomDataset(csv_file=test_csv, train=False, lens=lens, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    return dataloaders


# ----------------------------
# 训练和验证
# ----------------------------

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device='cuda',
                checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)  # (batch, 6000, 1)
                labels = labels.to(device)  # (batch, num_classes)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # (batch, num_classes)
                    loss = criterion(outputs, torch.argmax(labels, dim=1))

                    # 反向传播 + 优化，仅在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == torch.argmax(labels, dim=1))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                print("Best model saved")

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    writer.close()
    return model


# ----------------------------
# 预测和保存结果
# ----------------------------

def predict(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # (batch, num_classes)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions


def save_predictions(predictions, test_csv, output_file='predictions.csv'):
    test_data = pd.read_csv(test_csv)
    predictions_df = pd.DataFrame({
        'SampleID': test_data.index,  # 假设每一行是一个样本
        'PredictedClass': predictions
    })
    predictions_df.to_csv(output_file, index=False)
    print(f"预测完成，结果已保存为 '{output_file}'")


# ----------------------------
# 主训练流程
# ----------------------------

def main():
    # 参数设置
    train_csv = "train.csv"
    test_csv = "test.csv"
    batch_size = 32
    lens = 640
    num_classes = 10
    num_epochs = 10
    embed_dim = 32
    window_size = 4
    num_heads = 2
    focal_level = 3
    focal_x = [3, 1, 1, 1]
    focal_factor = 2
    cardinality = 8
    mlp_ratio = 4.
    drop = 0.3
    drop_path = 0.1
    checkpoint_dir = 'checkpoints'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取 DataLoader
    dataloaders = get_dataloaders(train_csv, test_csv, batch_size=batch_size, lens=lens, num_classes=num_classes)

    # 初始化模型
    model = CustomModel(input_dim=6000, num_classes=num_classes, embed_dim=embed_dim, window_size=window_size,
                        num_heads=num_heads, focal_level=focal_level, focal_x=focal_x, focal_factor=focal_factor,
                        cardinality=cardinality, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device,
                        checkpoint_dir=checkpoint_dir)

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存为 '{final_model_path}'")

    # 预测
    test_dataset = CustomDataset(csv_file=test_csv, train=False, lens=lens, num_classes=num_classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    predictions = predict(model, test_loader, device=device)

    # 保存预测结果
    save_predictions(predictions, test_csv, output_file='predictions.csv')


if __name__ == "__main__":
    main()
