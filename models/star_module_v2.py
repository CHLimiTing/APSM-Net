"""
STAR模块v2版本
基于SOFTS模型的STar Aggregate-Redistribute模块，用于替换AMD中的DDI模块
实现高效的多变量通道间信息交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STAR_v2(nn.Module):
    """
    STar Aggregate-Redistribute Module v2
    基于SOFTS论文的STAR模块实现，用于替换DDI模块
    
    核心思想：
    1. 聚合(Aggregate): 将所有通道信息汇总到中心节点
    2. 重分配(Redistribute): 将全局信息广播回各个通道
    3. 星型交互: O(C)复杂度替代传统O(C²)通道交互
    """
    
    def __init__(self, input_shape, d_core=None, dropout=0.1, layernorm=True):
        """
        Args:
            input_shape: tuple (seq_len, feature_num)
            d_core: 核心表示维度，默认为seq_len
            dropout: dropout率
            layernorm: 是否使用层归一化
        """
        super(STAR_v2, self).__init__()
        
        self.seq_len = input_shape[0]  # L
        self.feature_num = input_shape[1]  # F
        self.d_series = self.seq_len  # 将序列长度作为series维度
        self.d_core = d_core if d_core is not None else self.seq_len
        
        self.layernorm = layernorm
        
        # STAR模块的四个线性变换层
        # gen1: 初步特征变换
        self.gen1 = nn.Linear(self.d_series, self.d_series)
        # gen2: 生成核心表示
        self.gen2 = nn.Linear(self.d_series, self.d_core)
        # gen3: 融合局部和全局信息
        self.gen3 = nn.Linear(self.d_series + self.d_core, self.d_series)
        # gen4: 最终输出变换
        self.gen4 = nn.Linear(self.d_series, self.d_series)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 移除内部层归一化，与原始SOFTS保持一致

    def forward(self, x):
        """
        前向传播
        Args:
            x: tensor [batch_size, feature_num, seq_len]
        Returns:
            output: tensor [batch_size, feature_num, seq_len]
        """
        batch_size, channels, d_series = x.shape
        
        # 直接使用输入，层归一化由外部EncoderLayer处理
        x_normed = x
        
        # 阶段1: 聚合(Aggregate) - 生成全局核心表示
        
        # 步骤1: 初步特征变换 (MLP₁)
        combined_mean = F.gelu(self.gen1(x_normed))
        combined_mean = self.dropout(combined_mean)
        
        # 步骤2: 生成核心表示
        combined_mean = self.gen2(combined_mean)
        
        # 步骤3: 随机池化 (Stochastic Pooling)
        if self.training:
            # 训练模式: 随机采样
            ratio = F.softmax(combined_mean, dim=1)  # [B, C, d_core]
            ratio = ratio.permute(0, 2, 1)  # [B, d_core, C]
            ratio = ratio.reshape(-1, channels)  # [B*d_core, C]
            
            # 多项式采样
            indices = torch.multinomial(ratio, 1)  # [B*d_core, 1]
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  # [B, 1, d_core]
            
            # 根据采样索引选择通道
            combined_mean = torch.gather(combined_mean, 1, indices)  # [B, 1, d_core]
            combined_mean = combined_mean.repeat(1, channels, 1)  # [B, C, d_core]
        else:
            # 推理模式: 加权平均
            weight = F.softmax(combined_mean, dim=1)  # [B, C, d_core]
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)  # [B, 1, d_core]
            combined_mean = combined_mean.repeat(1, channels, 1)  # [B, C, d_core]
        
        # 阶段2: 重分配与融合(Redistribute & Fuse)
        
        # 步骤4: 核心广播与拼接
        combined_mean_cat = torch.cat([x_normed, combined_mean], dim=-1)  # [B, C, d_series + d_core]
        
        # 步骤5: 信息融合 (MLP₂)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout(combined_mean_cat)
        combined_mean_cat = self.gen4(combined_mean_cat)  # [B, C, d_series]
        
        # 步骤6: 直接输出（原始SOFTS没有残差连接）
        output = combined_mean_cat
        
        return output


class MultiSTAR_v2(nn.Module):
    """
    多层STAR模块，对应原DDI的多block设计
    """
    
    def __init__(self, input_shape, n_block=1, d_core=None, dropout=0.1, layernorm=True):
        """
        Args:
            input_shape: tuple (seq_len, feature_num)
            n_block: STAR模块层数
            d_core: 核心表示维度
            dropout: dropout率
            layernorm: 是否使用层归一化
        """
        super(MultiSTAR_v2, self).__init__()
        
        self.n_block = n_block
        
        # 创建多个STAR模块
        self.star_blocks = nn.ModuleList([
            STAR_v2(input_shape, d_core=d_core, dropout=dropout, layernorm=layernorm)
            for _ in range(n_block)
        ])
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: tensor [batch_size, feature_num, seq_len]
        Returns:
            output: tensor [batch_size, feature_num, seq_len]
        """
        output = x
        
        # 逐层通过STAR模块
        for star_block in self.star_blocks:
            output = star_block(output)
        
        return output
