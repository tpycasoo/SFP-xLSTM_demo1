"""
================================================================================
SFP-xLSTM: Pseudocode and Algorithm Description
================================================================================

Paper: SFP-xLSTM: A fusion fault diagnosis framework of sparse focus 
modulation and xLSTM for vibration signals

Authors: Yubo Guan, Peng Li, Aiying Zhao, Shilin Wang
Institution: Lanzhou Jiaotong University
================================================================================
"""

# ==============================================================================
# Algorithm 1: GADF Encoding
# ==============================================================================
"""
算法 1: Gramian Angular Difference Field (GADF) 编码

输入: 一维振动信号 S = {x₁, x₂, ..., xₙ}
输出: 二维 GADF 图像 G ∈ R^(256×256)

步骤:
1. 归一化 (公式 1):
   FOR i = 1 TO n:
       x̃ᵢ = 2 * (xᵢ - min(S)) / (max(S) - min(S)) - 1
   END FOR

2. 极坐标转换 (公式 2):
   FOR i = 1 TO n:
       θᵢ = arccos(x̃ᵢ)    // 角度
       rᵢ = i / n          // 半径
   END FOR

3. GADF 矩阵计算 (公式 3):
   FOR i = 1 TO n:
       FOR j = 1 TO n:
           G[i,j] = sin(θᵢ - θⱼ)
       END FOR
   END FOR

4. 转换为灰度图:
   G_gray = (G + 1) / 2 * 255
   
5. 调整图像大小至 256×256

RETURN G_gray
"""

PSEUDOCODE_GADF = """
ALGORITHM GADF_Encoding(signal S):
    // Step 1: Normalization to [-1, 1]
    x_normalized = 2 * (S - min(S)) / (max(S) - min(S)) - 1
    
    // Step 2: Polar coordinate transformation
    theta = arccos(x_normalized)
    
    // Step 3: Compute GADF matrix
    // G[i,j] = sin(θᵢ - θⱼ) = sin(θᵢ)cos(θⱼ) - cos(θᵢ)sin(θⱼ)
    sin_theta = sin(theta)
    cos_theta = x_normalized  // cos(arccos(x)) = x
    
    G = outer_product(sin_theta, cos_theta) - outer_product(cos_theta, sin_theta)
    
    // Step 4: Convert to grayscale
    G_gray = ((G + 1) / 2) * 255
    
    // Step 5: Resize to target size
    G_resized = resize(G_gray, (256, 256))
    
    RETURN G_resized
"""

# ==============================================================================
# Algorithm 2: Sparse Focal Point Modulation (SFPM)
# ==============================================================================
"""
算法 2: 稀疏焦点调制模块 (SFPM)

输入: 特征图 X ∈ R^(B×C×H×W)
输出: 增强特征图 X_out ∈ R^(B×C'×H×W)

参数 (来自论文表4):
- 焦点层数 L = 3
- 基础核大小 k₀ = 3
- 膨胀率增长因子 β = 2
- 稀疏率 ρ = 0.3

步骤:
1. 特征分解 (公式 6, 7):
   F = Conv1×1(X)  // 投影到 2C+L+1 通道
   [Q, K, G] = Split(F)  // Q: 查询, K: 上下文, G: 门控权重

2. 重要性评估与稀疏化 (公式 11, 12, 13):
   M = σ(Conv3×3(ReLU(Conv1×1(K))))  // 重要性掩码
   k = max(⌊ρ·H·W⌋, K_min)  // 保留的特征数
   S = K ⊙ TopK(M, k)  // 稀疏化

3. 多尺度特征提取 (公式 8, 9):
   FOR l = 1 TO L:
       k_l = α·(l-1) + k₀  // 核大小
       d_l = β^(l-1)       // 膨胀率
       K^(l) = ReLU(BN(DSConv_{k_l,d_l}(K^(l-1))))
   END FOR

4. 门控聚合 (公式 14):
   Ĝ = Softmax(G, dim=1)
   K_agg = Σ_{l=1}^{L} Ĝ[l] ⊙ K^(l)

5. 全局上下文 (公式 15, 16):
   K_global = AdaptiveAvgPool2d(K^(L))
   K_final = K_agg + Ĝ[L+1] ⊙ K_global

6. 输出生成 (公式 17):
   V = Conv1×1(K_final)
   O = Dropout(Linear(Q ⊙ V))
   X_out = X + O

RETURN X_out
"""

PSEUDOCODE_SFPM = """
ALGORITHM SFPM(X, L=3, k0=3, beta=2, rho=0.3):
    // Step 1: Feature decomposition
    F = Conv1x1(X)  // Project to (2C + L + 1) channels
    Q, K, G = Split(F, [C, C, L+1])
    
    // Step 2: Importance assessment and sparsification
    M = Sigmoid(Conv3x3(ReLU(Conv1x1(K))))  // Importance mask
    k = max(floor(rho * H * W), K_min)
    topk_mask = TopK(M.flatten(), k)
    K_sparse = K * topk_mask.reshape(1, 1, H, W)
    
    // Step 3: Multi-scale feature extraction
    K_levels = []
    K_current = K_sparse
    FOR l = 1 TO L:
        kernel_size = alpha * (l - 1) + k0
        dilation = beta ^ (l - 1)
        K_current = ReLU(BatchNorm(DSConv(K_current, kernel_size, dilation)))
        K_levels.append(K_current)
    END FOR
    
    // Step 4: Gated aggregation
    G_normalized = Softmax(G, dim=1)
    K_agg = zeros_like(K_levels[0])
    FOR l = 0 TO L-1:
        K_agg = K_agg + G_normalized[:, l] * K_levels[l]
    END FOR
    
    // Step 5: Global context incorporation
    K_global = AdaptiveAvgPool2d(K_levels[-1], (1, 1))
    K_global = broadcast(K_global, (H, W))
    K_final = K_agg + G_normalized[:, -1] * K_global
    
    // Step 6: Output with residual connection
    V = Conv1x1(K_final)
    O = Dropout(Linear(Q * V))  // Element-wise multiplication
    X_out = X + O
    
    RETURN X_out
"""

# ==============================================================================
# Algorithm 3: Extended LSTM (xLSTM)
# ==============================================================================
"""
算法 3: 扩展长短期记忆网络 (xLSTM)

输入: 特征序列 F_seq ∈ R^(B×T×C)
输出: 时序特征 h ∈ R^(B×D_h)

参数 (来自论文表4):
- 隐藏维度 D_h = 256
- 记忆子空间数 N_s = 4
- Dropout率 = 0.2

包含两种单元:
- sLSTM: 标量存储 + 指数门控
- mLSTM: 矩阵存储 + 协方差更新规则
"""

PSEUDOCODE_SLSTM = """
ALGORITHM sLSTM_Cell(x_t, h_{t-1}, c_{t-1}, n_{t-1}):
    // Standard gating equations (公式 18)
    gates = W_ih * x_t + W_hh * h_{t-1}
    i_pre, f_pre, o_pre, c_tilde = Split(gates, 4)
    
    // Exponential gating (公式 19)
    i_t = exp(i_pre)                    // 指数输入门
    f_t = sigmoid(f_pre + b_f)          // 遗忘门
    o_t = sigmoid(o_pre)                // 输出门
    c_tilde = tanh(c_tilde)             // 候选状态
    
    // Cell state update (公式 19)
    c_t = f_t * c_{t-1} + i_t * c_tilde
    
    // Normalizer update (公式 20)
    n_t = f_t * n_{t-1} + i_t
    
    // Output
    h_t = o_t * tanh(c_t / n_t)
    
    RETURN h_t, c_t, n_t
"""

PSEUDOCODE_MLSTM = """
ALGORITHM mLSTM_Cell(x_t, h_{t-1}, C_{t-1}, n_{t-1}):
    // Query, Key, Value projections
    q_t = W_q * x_t
    k_t = W_k * x_t
    v_t = W_v * x_t
    
    // Exponential gating
    i_t = exp(W_i * x_t)
    f_t = sigmoid(W_f * x_t)
    
    // Matrix memory update (公式 21)
    // C_{t+1} = f_t * C_t + i_t * v_t * k_t^T
    C_t = f_t * C_{t-1} + i_t * outer_product(v_t, k_t)
    
    // Normalizer update
    n_t = f_t * n_{t-1} + i_t
    
    // Retrieve from memory
    h_tilde = (C_t * q_t) / n_t
    
    // Output
    o_t = sigmoid(W_o * x_t)
    h_t = o_t * tanh(h_tilde)
    
    RETURN h_t, C_t, n_t
"""

PSEUDOCODE_XLSTM = """
ALGORITHM xLSTM(sequence, N_s=4):
    // Initialize states
    h, c, n = zeros(hidden_size)
    subspace_states = [zeros(hidden_size/N_s) for _ in range(N_s)]
    
    // Set forget gate biases (公式 23)
    FOR s = 1 TO N_s:
        b_f^(s) = -log(2^(s-1))
    END FOR
    
    // Process sequence
    FOR t = 1 TO seq_length:
        x_t = sequence[:, t, :]
        
        // Process through alternating sLSTM/mLSTM layers
        FOR layer in layers:
            IF layer is sLSTM:
                h, c, n = sLSTM_Cell(x_t, h, c, n)
            ELSE:
                h, C, n = mLSTM_Cell(x_t, h, C, n)
            x_t = Dropout(h) + x_t  // Residual connection
        END FOR
        
        // Multi-scale memory subspaces (公式 22)
        subspace_outputs = []
        FOR s = 1 TO N_s:
            h_s = sLSTM_Cell(x_t_chunk[s], subspace_states[s])
            subspace_outputs.append(h_s)
        END FOR
        
        // Attention-based memory fusion (公式 24)
        // α_t^(s) = softmax(w_a^T * tanh(W_a * c_t^(s)))
        attention_scores = []
        FOR s = 1 TO N_s:
            score = w_a^T * tanh(W_a * subspace_outputs[s])
            attention_scores.append(score)
        END FOR
        alpha = Softmax(attention_scores)
        
        // Fused output
        // c_fused = Σ α_t^(s) * c_t^(s)
        h_fused = sum(alpha[s] * subspace_outputs[s] for s in range(N_s))
        h_final = repeat(h_fused, N_s)
    END FOR
    
    RETURN LayerNorm(h_final)
"""

# ==============================================================================
# Algorithm 4: Complete SFP-xLSTM Forward Pass
# ==============================================================================
"""
算法 4: SFP-xLSTM 完整前向传播

输入: 一维振动信号 x ∈ R^n
输出: 故障类别预测 ŷ ∈ R^K

完整流程 (公式 35):
ŷ = Softmax(W_cls · xLSTM(SFPM(GADF(x))) + b_cls)
"""

PSEUDOCODE_COMPLETE = """
ALGORITHM SFP_xLSTM_Forward(signal x):
    // Step 1: GADF Encoding
    // Convert 1D signal to 2D image
    X_gadf = GADF_Encoding(x)  // (1, 256, 256)
    
    // Step 2: SFPM Feature Extraction
    // Extract multi-scale features with sparsification
    X_features = SFPM(X_gadf)  // (C, H, W)
    
    // Step 3: Adaptive Pooling (公式 34)
    // F_pool = AdaptiveAvgPool2d(X_out, (P, P))
    // F_seq = Reshape(F_pool, [B, P², C])
    X_pooled = AdaptiveAvgPool2d(X_features, (P, P))  // (C, P, P)
    X_seq = Reshape(X_pooled, (P*P, C))  // (P², C)
    
    // Step 4: xLSTM Temporal Modeling
    h = xLSTM(X_seq)  // (hidden_size,)
    
    // Step 5: Classification (公式 35)
    // ŷ = Softmax(W_cls · h + b_cls)
    logits = W_cls * h + b_cls  // (num_classes,)
    y_pred = Softmax(logits)
    
    RETURN y_pred
"""

# ==============================================================================
# Algorithm 5: Training Procedure
# ==============================================================================
"""
算法 5: 训练过程

训练参数 (来自论文表4):
- Batch Size: 32
- Learning Rate: 0.003
- Epochs: 150
- Optimizer: Adam

损失函数 (公式 36):
L = L_CE + λ₁·L_sparse + λ₂·L_temporal
"""

PSEUDOCODE_TRAINING = """
ALGORITHM Train_SFP_xLSTM(dataset, num_epochs=150, batch_size=32, lr=0.003):
    // Initialize model
    model = SFP_xLSTM(num_classes=K)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    // Training loop
    FOR epoch = 1 TO num_epochs:
        model.train()
        total_loss = 0
        
        FOR batch in DataLoader(dataset, batch_size):
            x, y = batch
            
            // Forward pass
            y_pred = model(x)
            
            // Compute loss (公式 36)
            L_CE = CrossEntropyLoss(y_pred, y)
            L_sparse = mean(|importance_maps|)  // L1 sparsity
            L_temporal = mean((features[:, 1:] - features[:, :-1])^2)
            
            L = L_CE + λ₁ * L_sparse + λ₂ * L_temporal
            
            // Backward pass
            optimizer.zero_grad()
            L.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += L.item()
        END FOR
        
        // Validation
        val_loss, val_acc = Validate(model, val_loader)
        scheduler.step(val_loss)
        
        // Early stopping check
        IF val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model)
            patience_counter = 0
        ELSE:
            patience_counter += 1
            IF patience_counter >= 20:
                BREAK
        END IF
    END FOR
    
    RETURN model
"""

# ==============================================================================
# Parameter Summary
# ==============================================================================
"""
模型参数总结 (来自论文表4)

模块          参数名称                    值
============================================================
GAF          图像大小                    256 × 256
             编码方法                    GADF

SFPM         焦点层数 (L)               3
             基础核大小 (k)              3
             膨胀率增长因子              2
             稀疏率 (ρ)                 0.3

xLSTM        隐藏维度                    256
             记忆子空间数 (N_s)          4
             Dropout率                  0.2

训练         Batch Size                 32
             学习率                      0.003
             训练轮数                    150
             优化器                      Adam
============================================================

最优稀疏率范围 (来自论文公式27): [0.15, 0.35]
最优隐藏维度 (来自论文表3): 256
"""

if __name__ == '__main__':
    print("SFP-xLSTM Pseudocode and Algorithms")
    print("=" * 60)
    print("\n1. GADF Encoding:")
    print(PSEUDOCODE_GADF)
    print("\n2. SFPM Module:")
    print(PSEUDOCODE_SFPM)
    print("\n3. sLSTM Cell:")
    print(PSEUDOCODE_SLSTM)
    print("\n4. mLSTM Cell:")
    print(PSEUDOCODE_MLSTM)
    print("\n5. xLSTM Module:")
    print(PSEUDOCODE_XLSTM)
    print("\n6. Complete Forward Pass:")
    print(PSEUDOCODE_COMPLETE)
    print("\n7. Training Procedure:")
    print(PSEUDOCODE_TRAINING)
