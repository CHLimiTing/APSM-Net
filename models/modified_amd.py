# import torch
# import torch.nn as nn
# from models.common import RevIN, DDI
# from models.mamba import SimpleMambaPredictor, MambaPredictor
# from models.msp import MultiScalePatchModule
#
#
# class ModifiedAMD(nn.Module):
#     """
#     修改后的AMD模型
#     - 使用MultiScalePatchModule替代MDM
#     - 使用MambaPredictor替代AMS
#     - 保持DDI模块不变
#     """
#
#     def __init__(self, input_shape, pred_len, n_block, dropout, patch, k, c, alpha,
#                  target_slice, norm=True, layernorm=True, use_simple_mamba=False):
#         super(ModifiedAMD, self).__init__()
#
#         self.target_slice = target_slice
#         self.norm = norm
#
#         # RevIN归一化
#         if self.norm:
#             self.rev_norm = RevIN(input_shape[-1])
#
#         # 多尺度Patch模块 (替代MDM)
#         self.multi_scale_patch = MultiScalePatchModule(
#             input_shape,
#             scales=[256, 64, 16, 4, 1],
#             layernorm=layernorm,
#             dropout=dropout
#         )
#
#         # DDI模块保持不变
#         self.fc_blocks = nn.ModuleList([
#             DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
#             for _ in range(n_block)
#         ])
#
#         # Mamba预测模块 (替代AMS)
#         if use_simple_mamba:
#             self.mamba_predictor = SimpleMambaPredictor(
#                 input_shape, pred_len, dropout=dropout
#             )
#         else:
#             self.mamba_predictor = MambaPredictor(
#                 input_shape, pred_len, dropout=dropout
#             )
#
#     def forward(self, x):
#         """
#         x: [batch_size, seq_len, feature_num]
#         return: prediction [batch_size, pred_len, target_features], loss
#         """
#         # RevIN归一化
#         if self.norm:
#             x = self.rev_norm(x, 'norm')
#
#         # 转置为 [batch_size, feature_num, seq_len]
#         x = torch.transpose(x, 1, 2)
#
#         # 多尺度Patch处理
#         multi_scale_features = self.multi_scale_patch(x)
#
#         # DDI处理
#         ddi_input = multi_scale_features.clone()  # 使用多尺度特征作为DDI输入
#         for fc_block in self.fc_blocks:
#             ddi_input = fc_block(ddi_input)
#
#         # Mamba预测 (融合多尺度特征和DDI输出)
#         prediction = self.mamba_predictor(multi_scale_features, ddi_input)
#
#         # 转置回 [batch_size, pred_len, feature_num]
#         prediction = torch.transpose(prediction, 1, 2)
#
#         # RevIN反归一化
#         if self.norm:
#             prediction = self.rev_norm(prediction, 'denorm', self.target_slice)
#
#         # 如果有目标切片，只返回目标特征
#         if self.target_slice:
#             prediction = prediction[:, :, self.target_slice]
#
#         # 返回预测结果和0损失 (因为不再使用MoE，没有负载均衡损失)
#         return prediction, torch.tensor(0.0).to(prediction.device)