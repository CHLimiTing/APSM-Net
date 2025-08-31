"""
LightGTS核心功能模块
实现基于周期性的动态多尺度patch处理功能
"""

import torch
import torch.nn.functional as F


def resize(x, target_patch_len):
    """
    将patch调整到目标长度
    Args:
        x: tensor [bs x num_patch x n_vars x patch_len]
        target_patch_len: 目标patch长度
    Returns:
        resized tensor [bs x num_patch x n_vars x target_patch_len]
    """
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs * num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear', align_corners=False)
    return x.reshape(bs, num_patch, n_vars, target_patch_len)


def FFT_for_Period(x, k=1):
    """
    使用FFT检测序列的主要周期
    Args:
        x: tensor [B, T, C] - 批次大小、时间长度、通道数
        k: 返回前k个重要周期
    Returns:
        period: 检测到的周期长度数组
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # 通过幅度找到周期
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0:2] = 0  # 去除DC分量和最低频
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period


def create_patch(xb, patch_len, stride):
    """
    创建patch
    Args:
        xb: tensor [bs x seq_len x n_vars]
        patch_len: patch长度
        stride: 步长
    Returns:
        patches: [bs x num_patch x n_vars x patch_len]
        num_patch: patch数量
    """
    xb = padding(xb, patch_len)
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len
    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


def padding(x, patch_len, order=True):
    """
    对序列进行padding
    Args:
        x: tensor [bs x seq_len x n_vars]
        patch_len: patch长度
        order: True表示在前面padding，False表示在后面padding
    Returns:
        padded tensor
    """
    padding_len = x.shape[1] % patch_len
    if padding_len != 0:
        padding_len = patch_len - padding_len
        padding = torch.zeros([x.shape[0], int(padding_len), x.shape[2]]).to(x.device)
        if order:
            x = torch.cat((padding, x), dim=1)
        else:
            x = torch.cat((x, padding), dim=1)
    return x


def reconstruct_from_patches(patches, target_seq_len):
    """
    从patches重构回序列
    Args:
        patches: tensor [bs x num_patch x n_vars x patch_len]
        target_seq_len: 目标序列长度
    Returns:
        reconstructed sequence [bs x target_seq_len x n_vars]
    """
    bs, num_patch, n_vars, patch_len = patches.shape
    # 展平patches
    flattened = patches.reshape(bs, -1, n_vars)  # [bs, num_patch*patch_len, n_vars]
    
    # 使用插值调整到目标长度
    # 需要转置以便插值
    flattened = flattened.permute(0, 2, 1)  # [bs, n_vars, num_patch*patch_len]
    reconstructed = F.interpolate(flattened, size=target_seq_len, mode='linear', align_corners=False)
    reconstructed = reconstructed.permute(0, 2, 1)  # [bs, target_seq_len, n_vars]
    
    return reconstructed
