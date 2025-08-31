"""
基于LightGTS和STAR改进的AMD模型v2版本
使用PeriodicalPatchMixer替换MDM模块，使用STAR模块替换DDI模块
"""

import torch
import torch.nn as nn

from models.common import RevIN
from models.periodical_patch_mixer import PeriodicalPatchMixer
from models.star_module_v2 import MultiSTAR_v2
from models.tsmoe import AMS


class AMD_LightGTS_v2(nn.Module):
    """
    Implementation of AMD with LightGTS PeriodicalPatchMixer and SOFTS STAR modules.
    第二版本：使用PeriodicalPatchMixer + STAR模块的组合
    """

    def __init__(self, input_shape, pred_len, dropout, 
                 top_k=3, target_patch_len=16, d_core=None, alpha=0.0, 
                 target_slice=None, norm=True, layernorm=True, e_layers=1):
        """
        Args:
            input_shape: tuple (seq_len, feature_num)
            pred_len: 预测长度
            dropout: dropout率
            top_k: PeriodicalPatchMixer中的周期数量
            target_patch_len: PeriodicalPatchMixer中的目标patch长度
            d_core: STAR模块核心表示维度，默认为seq_len
            alpha: 保留参数，用于兼容性（STAR中不使用）
            target_slice: 目标切片
            norm: 是否使用RevIN归一化
            layernorm: 是否使用层归一化
            e_layers: STAR模块层数（对应原始SOFTS的e_layers）
        """
        super(AMD_LightGTS_v2, self).__init__()

        self.target_slice = target_slice
        self.norm = norm
        self.alpha = alpha  # 保留以保持接口兼容

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

        # 使用MultiSTAR_v2替换原始的DDI模块
        # 注意：这里使用e_layers而不是n_block来控制STAR层数
        self.star_blocks = MultiSTAR_v2(
            input_shape,
            n_block=e_layers,  # 使用e_layers控制STAR层数
            d_core=d_core,
            dropout=dropout,
            layernorm=layernorm
        )

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

        # STAR模块处理（替换DDI）
        x = self.star_blocks(x)

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


def create_amd_lightgts_v2_model(args, n_feature, target_slice=None):
    """
    便捷函数：创建AMD_LightGTS_v2模型实例
    Args:
        args: 参数对象，包含模型配置
        n_feature: 特征数量
        target_slice: 目标切片
    Returns:
        AMD_LightGTS_v2模型实例
    """
    return AMD_LightGTS_v2(
        input_shape=(args.seq_len, n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        n_block=args.n_block,
        top_k=getattr(args, 'top_k', 3),  # 默认值3
        target_patch_len=getattr(args, 'target_patch_len', 16),  # 默认值16
        d_core=getattr(args, 'd_core', None),  # 默认值None（使用seq_len）
        alpha=args.alpha,  # 保持兼容性
        target_slice=target_slice,
        norm=args.norm,
        layernorm=args.layernorm
    )
