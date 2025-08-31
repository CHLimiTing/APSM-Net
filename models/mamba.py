# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 注意：这里假设使用mamba-ssm库，如果没有安装需要先安装
# # pip install mamba-ssm
# try:
#     from mamba_ssm import Mamba
#
#     MAMBA_AVAILABLE = True
# except ImportError:
#     print("Warning: mamba-ssm not available, using LSTM as fallback")
#     MAMBA_AVAILABLE = False
#
#
# class MambaPredictor(nn.Module):
#     """
#     Mamba预测模块，替代原来的AMS模块
#     接收多尺度特征和DDI输出，进行序列到序列的预测
#     """
#
#     def __init__(self, input_shape, pred_len, d_state=64, d_conv=4, expand=2, dropout=0.1):
#         super(MambaPredictor, self).__init__()
#
#         self.seq_len = input_shape[0]  # 512
#         self.feature_num = input_shape[1]
#         self.pred_len = pred_len
#         self.d_model = self.seq_len  # Mamba的模型维度
#
#         if MAMBA_AVAILABLE:
#             # 使用Mamba层进行序列建模
#             self.mamba_layer = Mamba(
#                 d_model=self.d_model,
#                 d_state=d_state,
#                 d_conv=d_conv,
#                 expand=expand
#             )
#         else:
#             # LSTM作为fallback
#             self.lstm = nn.LSTM(
#                 input_size=self.d_model,
#                 hidden_size=self.d_model,
#                 num_layers=2,
#                 batch_first=True,
#                 dropout=dropout
#             )
#
#         # 渐进式压缩：序列长度映射 512 -> pred_len
#         self.seq_mapper = nn.Sequential(
#             nn.Linear(self.seq_len, self.seq_len // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.seq_len // 2, pred_len)
#         )
#
#         # 输出层归一化
#         self.output_norm = nn.LayerNorm(pred_len)
#
#     def forward(self, multi_scale_features, ddi_output):
#         """
#         multi_scale_features: [B, F, L] 多尺度特征
#         ddi_output: [B, F, L] DDI模块输出
#         return: [B, F, pred_len] 预测结果
#         """
#         # Element-wise addition融合两个输入
#         fused_features = multi_scale_features + ddi_output  # [B, F, L]
#
#         batch_size, feature_num, seq_len = fused_features.shape
#         output = torch.zeros(batch_size, feature_num, self.pred_len).to(fused_features.device)
#
#         # 对每个特征维度分别处理
#         for i in range(feature_num):
#             feature_seq = fused_features[:, i, :]  # [B, L]
#
#             if MAMBA_AVAILABLE:
#                 # 使用Mamba进行序列建模
#                 mamba_output = self.mamba_layer(feature_seq)  # [B, L]
#             else:
#                 # 使用LSTM作为fallback
#                 lstm_output, _ = self.lstm(feature_seq.unsqueeze(-1))  # [B, L, 1]
#                 mamba_output = lstm_output.squeeze(-1)  # [B, L]
#
#             # 序列长度映射：L -> pred_len
#             pred_seq = self.seq_mapper(mamba_output)  # [B, pred_len]
#
#             # 归一化
#             pred_seq = self.output_norm(pred_seq)
#
#             output[:, i, :] = pred_seq
#
#         return output
#
#
# class SimpleMambaPredictor(nn.Module):
#     """
#     简化版Mamba预测模块，不依赖外部库
#     使用多层MLP + 残差连接模拟Mamba的长序列建模能力
#     """
#
#     def __init__(self, input_shape, pred_len, hidden_dim=512, num_layers=4, dropout=0.1):
#         super(SimpleMambaPredictor, self).__init__()
#
#         self.seq_len = input_shape[0]
#         self.feature_num = input_shape[1]
#         self.pred_len = pred_len
#         self.num_layers = num_layers
#
#         # 多层MLP模拟序列建模
#         self.mlp_layers = nn.ModuleList()
#         for i in range(num_layers):
#             layer = nn.Sequential(
#                 nn.Linear(self.seq_len, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, self.seq_len),
#                 nn.Dropout(dropout)
#             )
#             self.mlp_layers.append(layer)
#
#         # 渐进式压缩
#         self.seq_mapper = nn.Sequential(
#             nn.Linear(self.seq_len, self.seq_len // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.seq_len // 2, pred_len)
#         )
#
#         self.output_norm = nn.LayerNorm(pred_len)
#
#     def forward(self, multi_scale_features, ddi_output):
#         """
#         multi_scale_features: [B, F, L]
#         ddi_output: [B, F, L]
#         return: [B, F, pred_len]
#         """
#         # Element-wise addition融合
#         fused_features = multi_scale_features + ddi_output
#
#         batch_size, feature_num, seq_len = fused_features.shape
#         output = torch.zeros(batch_size, feature_num, self.pred_len).to(fused_features.device)
#
#         # 对每个特征维度分别处理
#         for i in range(feature_num):
#             feature_seq = fused_features[:, i, :]  # [B, L]
#
#             # 多层MLP + 残差连接
#             current = feature_seq
#             for mlp_layer in self.mlp_layers:
#                 current = current + mlp_layer(current)
#
#             # 序列长度映射
#             pred_seq = self.seq_mapper(current)
#             pred_seq = self.output_norm(pred_seq)
#
#             output[:, i, :] = pred_seq
#
#         return output