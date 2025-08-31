# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class MultiScalePatchModule(nn.Module):
#     """
#     多尺度Patch模块，替代原来的MDM模块
#     使用Top-down递归方式处理多个尺度的patch
#     """
#
#     def __init__(self, input_shape, scales=[256, 64, 16, 4, 1], layernorm=True, dropout=0.1):
#     # def __init__(self, input_shape, scales=[512, 128, 32, 8, 2], layernorm=True, dropout=0.1):
#         super(MultiScalePatchModule, self).__init__()
#
#         self.seq_len = input_shape[0]  # 512
#         self.feature_num = input_shape[1]
#         self.scales = scales  # [256, 64, 16, 4, 1] 从粗到细
#         self.layernorm = layernorm
#
#         # 为每个尺度创建patch处理层
#         self.patch_processors = nn.ModuleList()
#         self.linear_projectors = nn.ModuleList()
#
#         for i, scale in enumerate(scales):
#             patch_count = self.seq_len // scale
#
#             # 线性投影层：将patch_count映射到seq_len
#             projector = nn.Linear(patch_count, self.seq_len)
#             self.linear_projectors.append(projector)
#
#             # 复杂处理层：MLP + 激活函数 + Dropout
#             if i < len(scales) - 1:  # 最后一层不需要处理
#                 hidden_dim = self.seq_len // 2  # 使用较小的hidden_dim
#                 processor = nn.Sequential(
#                     nn.Linear(self.seq_len, hidden_dim),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(hidden_dim, self.seq_len)
#                 )
#                 self.patch_processors.append(processor)
#
#         # LayerNorm
#         if self.layernorm:
#             self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[1])
#
#     def create_patches(self, x, patch_size):
#         """
#         创建指定大小的patches
#         x: [B, F, L]
#         patch_size: int
#         return: [B, F, patch_count]
#         """
#         batch_size, feature_num, seq_len = x.shape
#         patch_count = seq_len // patch_size
#
#         # 重塑为patches并取平均
#         x_reshaped = x.view(batch_size, feature_num, patch_count, patch_size)
#         patches = x_reshaped.mean(dim=-1)  # [B, F, patch_count]
#
#         return patches
#
#     def forward(self, x):
#         """
#         x: [B, F, L] where L = seq_len = 512
#         return: [B, F, L] 多尺度特征融合后的结果
#         """
#         if self.layernorm:
#             x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
#
#         # Top-down递归处理
#         # 从最粗尺度开始
#         coarsest_scale = self.scales[0]  # 256
#         patches = self.create_patches(x, coarsest_scale)  # [B, F, 2]
#
#         # 投影到原始长度
#         output = self.linear_projectors[0](patches)  # [B, F, 512]
#
#         # 递归处理每个尺度
#         for i in range(1, len(self.scales)):
#             scale = self.scales[i]
#
#             # 创建当前尺度的patches
#             current_patches = self.create_patches(x, scale)
#
#             # 投影到原始长度
#             current_features = self.linear_projectors[i](current_patches)
#
#             # 残差连接
#             output = output + current_features
#
#             # 如果不是最后一层，通过处理层
#             if i < len(self.scales) - 1:
#                 output = output + self.patch_processors[i - 1](output)
#
#         return output