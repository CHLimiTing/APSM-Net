"""
PeriodicalPatchMixer模块
基于LightGTS周期性思想的动态多尺度patch处理模块，用于替换AMD中的MDM模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lightgts_utils import FFT_for_Period, create_patch, resize, reconstruct_from_patches


class PeriodicalPatchMixer(nn.Module):
    """
    基于周期性的动态多尺度patch混合器
    保持与原MDM模块完全相同的输入输出接口：[B, F, L] -> [B, F, L]
    """
    
    def __init__(self, input_shape, top_k=3, target_patch_len=16, layernorm=True, dropout=0.1):
        """
        Args:
            input_shape: tuple (seq_len, feature_num)
            top_k: 提取前k个重要周期
            target_patch_len: 统一的patch目标长度
            layernorm: 是否使用层归一化
            dropout: dropout率
        """
        super(PeriodicalPatchMixer, self).__init__()
        
        self.seq_len = input_shape[0]
        self.feature_num = input_shape[1]
        self.top_k = top_k
        self.target_patch_len = target_patch_len
        self.layernorm = layernorm
        
        # 层归一化
        if self.layernorm:
            self.norm_input = nn.BatchNorm1d(self.seq_len * self.feature_num)
            self.norm_output = nn.BatchNorm1d(self.seq_len * self.feature_num)
        
        # patch处理网络
        self.patch_processor = nn.Sequential(
            nn.Linear(self.target_patch_len, self.target_patch_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.target_patch_len * 2, self.target_patch_len),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多尺度融合权重
        self.fusion_weights = nn.Parameter(torch.ones(self.top_k))
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_num, self.feature_num * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_num * 2, self.feature_num)
        )
        
    def forward(self, x):
        """
        Args:
            x: tensor [batch_size, feature_num, seq_len]
        Returns:
            output: tensor [batch_size, feature_num, seq_len]
        """
        batch_size, feature_num, seq_len = x.shape
        
        # 输入层归一化
        if self.layernorm:
            x_normed = self.norm_input(torch.flatten(x, 1, -1)).reshape(x.shape)
        else:
            x_normed = x
        
        # 步骤1: 转换格式以适配FFT周期检测 [B, F, L] -> [B, L, F]
        x_for_fft = x_normed.permute(0, 2, 1)  # [batch_size, seq_len, feature_num]
        
        # 步骤2: 自适应周期提取
        try:
            periods = FFT_for_Period(x_for_fft, k=self.top_k)
            # 确保周期值有效且不超过序列长度
            periods = [max(4, min(p, seq_len // 2)) for p in periods if p > 0]
            if len(periods) == 0:
                periods = [seq_len // 4, seq_len // 8, seq_len // 16]  # 默认周期
            elif len(periods) < self.top_k:
                # 补充默认周期
                default_periods = [seq_len // 4, seq_len // 8, seq_len // 16]
                periods.extend([p for p in default_periods if p not in periods])
                periods = periods[:self.top_k]
        except:
            # 如果FFT失败，使用默认周期
            periods = [seq_len // 4, seq_len // 8, seq_len // 16]
        
        # 步骤3: 多周期patch创建与处理
        multi_scale_representations = []
        
        for i, period in enumerate(periods):
            try:
                # 创建patch
                patches, num_patch = create_patch(x_for_fft, patch_len=period, stride=period)
                # [batch_size, num_patch, feature_num, period]
                
                # 归一化到统一长度
                if period != self.target_patch_len:
                    patches = resize(patches, self.target_patch_len)
                # [batch_size, num_patch, feature_num, target_patch_len]
                
                # 处理patches
                original_shape = patches.shape
                patches_flat = patches.reshape(-1, self.target_patch_len)
                processed_patches = self.patch_processor(patches_flat)
                processed_patches = processed_patches.reshape(original_shape)
                
                # 重构回序列
                reconstructed = reconstruct_from_patches(processed_patches, seq_len)
                # [batch_size, seq_len, feature_num]
                
                multi_scale_representations.append(reconstructed)
                
            except Exception as e:
                # 如果处理失败，使用原始输入作为备选
                print(f"Warning: Period {period} processing failed: {e}")
                multi_scale_representations.append(x_for_fft)
        
        # 步骤4: 多尺度信息融合
        if len(multi_scale_representations) > 0:
            # 堆叠所有尺度的表示
            stacked_representations = torch.stack(multi_scale_representations, dim=0)
            # [num_scales, batch_size, seq_len, feature_num]
            
            # 加权平均融合
            fusion_weights_normalized = F.softmax(self.fusion_weights[:len(multi_scale_representations)], dim=0)
            fusion_weights_expanded = fusion_weights_normalized.view(-1, 1, 1, 1)
            
            fused_representation = torch.sum(stacked_representations * fusion_weights_expanded, dim=0)
            # [batch_size, seq_len, feature_num]
        else:
            fused_representation = x_for_fft
        
        # 步骤5: 转换回原始格式 [B, L, F] -> [B, F, L]
        fused_representation = fused_representation.permute(0, 2, 1)
        # [batch_size, feature_num, seq_len]
        
        # 步骤6: 输出投影和残差连接
        fused_transposed = fused_representation.permute(0, 2, 1)  # [B, L, F]
        projected = self.output_projection(fused_transposed)
        projected = projected.permute(0, 2, 1)  # [B, F, L]
        
        # 残差连接
        output = x + projected
        
        # 输出层归一化
        if self.layernorm:
            output = self.norm_output(torch.flatten(output, 1, -1)).reshape(output.shape)
        
        return output
