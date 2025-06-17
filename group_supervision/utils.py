import torch
import torch.nn.functional as F
import numpy as np


def _dot_similarity_dim1(x, y):
    v = torch.matmul(x.unsqueeze(1), y.unsqueeze(2))
    return v


def _dot_similarity_dim2(x, y):
    v = torch.tensordot(x.unsqueeze(1), y.t().unsqueeze(0), dims=2)
    return v


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=torch.bool)
    for i in range(batch_size):
        negative_mask[i, i] = False
        negative_mask[i, i + batch_size] = False
    return negative_mask


def _nan_to_zero(x):
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def pearson_correlation(y_true, y_pred):
    y_true = y_true - y_true.mean(dim=-1, keepdim=True)
    y_pred = y_pred - y_pred.mean(dim=-1, keepdim=True)
    
    numerator = (y_true * y_pred).sum(dim=-1)
    denominator = torch.sqrt((y_true ** 2).sum(dim=-1) * (y_pred ** 2).sum(dim=-1))
    
    corr = numerator / (denominator + 1e-8)
    corr = _nan_to_zero(corr)
    
    return corr
