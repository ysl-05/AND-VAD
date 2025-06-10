import torch as th
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence_loss(pred_features, z_sp, reduction='mean'):
    """
    Calculate KL divergence loss between predicted features and z_sp
    
    :param pred_features: List of predicted features from model [B, C, D, H, W]
    :param z_sp: Target features from encoder [B, C, D, H, W]
    :param reduction: Reduction method ('mean' or 'none')
    :return: KL divergence loss
    """
    total_loss = 0
    num_features = 0
    
    for feat in pred_features:
        # Reshape z_sp to match feature dimensions if needed
        if z_sp.shape[2:] != feat.shape[2:]:
            z_sp_resized = F.interpolate(z_sp, size=feat.shape[2:], mode='trilinear', align_corners=False)
        else:
            z_sp_resized = z_sp
            
        # Calculate KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(feat, dim=1),
            F.softmax(z_sp_resized, dim=1),
            reduction=reduction
        )
        
        if reduction == 'mean':
            total_loss += kl_loss
        else:
            total_loss += kl_loss.mean()
        num_features += 1
    
    return total_loss / num_features if num_features > 0 else total_loss

class DiffusionLoss(nn.Module):
    """
    Combined loss for diffusion model including denoising and feature alignment
    """
    def __init__(self, normal_align_weight=0.1):
        super().__init__()
        self.normal_align_weight = normal_align_weight
        
    def forward(self, pred_noise, target_noise, pred_features=None, z_sp=None):
        """
        Calculate combined loss
        
        :param pred_noise: Predicted noise [B, C, D, H, W]
        :param target_noise: Target noise [B, C, D, H, W]
        :param pred_features: List of predicted features from model
        :param z_sp: Target features from encoder
        :return: Combined loss
        """
        # Denoising loss
        denoising_loss = F.mse_loss(pred_noise, target_noise)
        
        # Feature alignment loss
        if pred_features is not None and z_sp is not None:
            align_loss = kl_divergence_loss(pred_features, z_sp)
            total_loss = denoising_loss + self.normal_align_weight * align_loss
        else:
            total_loss = denoising_loss
            
        return total_loss 