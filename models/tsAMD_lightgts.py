"""
基于LightGTS改进的AMD模型
使用PeriodicalPatchMixer替换原始的MDM模块
"""

import torch
import torch.nn as nn

from models.common import RevIN, DDI
from models.periodical_patch_mixer import PeriodicalPatchMixer
from models.tsmoe import AMS


class AMD_LightGTS(nn.Module):
    """
    Implementation of AMD with LightGTS-based PeriodicalPatchMixer.
    使用基于LightGTS周期性思想的PeriodicalPatchMixer替换原始MDM模块
    """

    def __init__(self, input_shape, pred_len, n_block, dropout, patch, 
                 top_k=3, target_patch_len=16, alpha=0.0, target_slice=None, 
                 norm=True, layernorm=True):
        """
        Args:
            input_shape: tuple (seq_len, feature_num)
            pred_len: 预测长度
            n_block: DDI块数量
            dropout: dropout率
            patch: DDI模块中的patch大小
            top_k: PeriodicalPatchMixer中的周期数量
            target_patch_len: PeriodicalPatchMixer中的目标patch长度
            alpha: DDI模块中的alpha参数
            target_slice: 目标切片
            norm: 是否使用RevIN归一化
            layernorm: 是否使用层归一化
        """
        super(AMD_LightGTS, self).__init__()

        self.target_slice = target_slice
        self.norm = norm

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        # 使用PeriodicalPatchMixer替换原始的MDM
        self.pastmixing = PeriodicalPatchMixer(
            input_shape, 
            top_k=top_k, 
            target_patch_len=target_patch_len, 
            layernorm=layernorm,
            dropout=dropout
        )

        # DDI块保持不变
        self.fc_blocks = nn.ModuleList([
            DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
            for _ in range(n_block)
        ])

        # AMS专家混合模块保持不变
        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        """
        前向传播
        Args:
            x: tensor [batch_size, seq_len, feature_num]
        Returns:
            output: tensor [batch_size, pred_len, target_features]
            moe_loss: MOE损失
        """
        # [batch_size, seq_len, feature_num]

        # 层归一化
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # 转置以适配后续模块 [batch_size, seq_len, feature_num] -> [batch_size, feature_num, seq_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        # 使用PeriodicalPatchMixer进行时间嵌入提取
        time_embedding = self.pastmixing(x)

        # DDI块处理
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # MOE专家混合：seq_len -> pred_len
        x, moe_loss = self.moe(x, time_embedding)

        # 转置回原始格式 [batch_size, feature_num, pred_len] -> [batch_size, pred_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        # 反归一化
        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        # 目标切片
        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss


def create_amd_lightgts_model(args, n_feature, target_slice=None):
    """
    便捷函数：创建AMD_LightGTS模型实例
    Args:
        args: 参数对象，包含模型配置
        n_feature: 特征数量
        target_slice: 目标切片
    Returns:
        AMD_LightGTS模型实例
    """
    return AMD_LightGTS(
        input_shape=(args.seq_len, n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        n_block=args.n_block,
        patch=args.patch,
        top_k=getattr(args, 'top_k', 3),  # 默认值3
        target_patch_len=getattr(args, 'target_patch_len', 16),  # 默认值16
        alpha=args.alpha,
        target_slice=target_slice,
        norm=args.norm,
        layernorm=args.layernorm
    )
