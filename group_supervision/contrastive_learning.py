import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from networks import Featurizer, Classifier, EncoderDecoder
import numpy as np


def _max_cross_corr(feats_1, feats_2):
    # feats_1: 1 x T(# time stamp)
    # feats_2: M(# aug) x T(# time stamp)
    feats_2 = feats_2.to(feats_1.dtype)
    feats_1 = feats_1 - feats_1.mean(dim=-1, keepdim=True)
    feats_2 = feats_2 - feats_2.mean(dim=-1, keepdim=True)

    min_N = min(feats_1.shape[-1], feats_2.shape[-1])
    padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
    feats_1_pad = F.pad(feats_1, (0, padded_N - feats_1.shape[-1]))
    feats_2_pad = F.pad(feats_2, (0, padded_N - feats_2.shape[-1]))

    feats_1_fft = fft.rfft(feats_1_pad)
    feats_2_fft = fft.rfft(feats_2_pad)
    X = feats_1_fft * torch.conj(feats_2_fft)

    power_norm = (feats_1.std(dim=-1, keepdim=True) * 
                 feats_2.std(dim=-1, keepdim=True))
    power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
    X = X / power_norm

    cc = fft.irfft(X) / (min_N - 1)
    max_cc = torch.max(cc, dim=-1)[0]

    return max_cc


def batched_max_cross_corr(x, y):
    """
    x: M(# aug) x T(# time stamp)
    y: M(# aug) x T(# time stamp)
    """
    dist = torch.stack([_max_cross_corr(x[i:i+1], y) for i in range(x.shape[0])])
    return dist


def normed_psd(x, fps, zero_pad=0, high_pass=0.25, low_pass=15):
    """ x: M(# aug) x T(# time stamp) """
    x = x - x.mean(dim=-1, keepdim=True)
    if zero_pad > 0:
        L = x.shape[-1]
        x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)))

    x = torch.abs(fft.rfft(x)) ** 2

    Fn = fps / 2
    freqs = torch.linspace(0., Fn, x.shape[-1], device=x.device)
    use_freqs = torch.logical_and(freqs >= high_pass, freqs <= low_pass)
    use_freqs = use_freqs.repeat(x.shape[0], 1)
    x = x[use_freqs].reshape(x.shape[0], -1)

    # Normalize PSD
    denom = torch.norm(x, dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    x = x / denom
    return x


def batched_normed_psd(x, y):
    """
    x: M(# aug) x T(# time stamp)
    y: M(# aug) x T(# time stamp)
    """
    return torch.matmul(normed_psd(x), normed_psd(y).t())


def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    # labels: bsz x M(#augs)
    # output: bsz x M(#augs) x M(#augs)
    if dist_fn == 'l1':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
    elif dist_fn == 'l2':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2
    elif dist_fn == 'sqrt':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 0.5
    else:
        raise NotImplementedError(f"`{dist_fn}` not implemented.")

    prob_mat = F.softmax(dist_mat / label_temperature, dim=-1)
    return prob_mat


class DisentangledAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(DisentangledAutoencoder, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.invariant_dim = bottleneck_dim // 2
        self.variant_dim = bottleneck_dim - self.invariant_dim
    
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, bottleneck_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # 编码到瓶颈层
        bottleneck = self.encoder(x)
        
        # 分离不变特征和可变特征
        invariant = bottleneck[:, :self.invariant_dim]
        variant = bottleneck[:, self.invariant_dim:]
        
        # 重建
        reconstructed = self.decoder(bottleneck)
        
        return invariant, variant, reconstructed, bottleneck


class FrameSequenceProcessor:
    def __init__(self, frame_length=5, overlap=2):
        self.frame_length = frame_length
        self.overlap = overlap
        
    def split_sequence(self, frames):
        T = frames.shape[0]
        sequences = []
        
        for i in range(0, T - self.frame_length + 1, self.frame_length - self.overlap):
            sequence = frames[i:i + self.frame_length]
            sequences.append(sequence)
            
        return torch.stack(sequences)
    
    def merge_sequences(self, sequences):
        N, L = sequences.shape[:2]
        overlap = self.frame_length - self.overlap
        T = (N - 1) * overlap + self.frame_length
        
        if sequences.shape[-1] == 3:
            output = torch.zeros((T, *sequences.shape[2:]), device=sequences.device)
        else:
            output = torch.zeros((T, *sequences.shape[2:]), device=sequences.device)
            
        for i in range(N):
            start = i * overlap
            if i == N - 1:
                output[start:] = sequences[i]
            else:
                output[start:start + self.frame_length] = sequences[i]
                
        return output


class Distangle(nn.Module):
    def __init__(self, hparams):
        super(Distangle, self).__init__()
        self.hparams = hparams
        
        # 序列处理参数
        self.frame_length = hparams.get("frame_length", 5)
        self.frame_overlap = hparams.get("frame_overlap", 2)
        
        # 特征提取器
        self.featurizer = Featurizer(self.frame_length)  # 修改为使用frame_length
        self.regressor = Classifier(self.featurizer.n_outputs, 1, False)
        
        # 自编码器
        self.autoencoder = DisentangledAutoencoder(
            input_dim=self.featurizer.n_outputs,
            hidden_dim=self.featurizer.n_outputs * 2,
            bottleneck_dim=self.hparams.get("bottleneck_dim", self.featurizer.n_outputs)
        )
        
        # 序列处理器
        self.sequence_processor = FrameSequenceProcessor(
            frame_length=self.frame_length,
            overlap=self.frame_overlap
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.featurizer.parameters()},
            {'params': self.regressor.parameters()},
            {'params': self.autoencoder.parameters()}
        ], lr=self.hparams["lr"])
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        # 内存库
        self.memory_bank = None
        self.memory_bank_size = self.hparams.get("memory_bank_size", 1024)
        self.memory_bank_ptr = 0
        
        # 序列增强组
        self.sequence_augmentation_groups = [
            # 时间域增强
            [random_reverse, arbitrary_speed_subsample],
            # 空间域增强
            [random_crop_resize, random_flip_left_right],
            [random_flip_up_down, random_rotation],
            # 外观域增强
            [random_grayscale_3d, random_brightness],
            [random_perspective, random_background_noise]
        ]

    def update_memory_bank(self, features):
        if self.memory_bank is None:
            self.memory_bank = torch.zeros(
                self.memory_bank_size, features.shape[1], 
                device=features.device
            )
        
        n = features.shape[0]
        self.memory_bank[self.memory_bank_ptr:self.memory_bank_ptr + n] = features
        self.memory_bank_ptr = (self.memory_bank_ptr + n) % self.memory_bank_size

    def compute_info_nce_loss(self, query, key, temperature=0.07):
        # 计算InfoNCE损失
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        # 计算与内存库中所有样本的相似度
        memory_sim = torch.matmul(query, self.memory_bank.t()) / temperature
        
        # 计算与正样本的相似度
        pos_sim = torch.sum(query * key, dim=1) / temperature
        
        # 合并正样本和负样本的相似度
        logits = torch.cat([pos_sim.unsqueeze(1), memory_sim], dim=1)
        
        # 标签：第一个样本是正样本
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        return self.criterion(logits, labels)

    def update(self, minibatches):
        all_x, all_speed = minibatches

        batch_size, num_augments = all_x.shape[0], all_x.shape[1]
        all_x = all_x.reshape(batch_size * num_augments, *all_x.shape[2:])

        all_labels = label_distance(all_speed[:, :num_augments // 2],
                                  all_speed[:, num_augments // 2:],
                                  self.hparams["label_dist_fn"],
                                  self.hparams["label_temperature"])

        self.optimizer.zero_grad()
        
        # 特征提取
        all_z = self.featurizer(all_x)
        all_z = all_z.reshape(batch_size, num_augments, -1)
        
        # 对比学习损失
        contrastive_loss = 0
        for feats, labels in zip(all_z, all_labels):
            feat_dist = globals()[self.hparams["feat_dist_fn"]](
                feats[:num_augments // 2], feats[num_augments // 2:])
            contrastive_loss += self.criterion(feat_dist, labels)
        contrastive_loss /= batch_size
        
        # 特征解耦损失
        disentangle_loss = 0
        reconstruction_loss = 0
        invariant_loss = 0
        variant_loss = 0
        
        for i in range(batch_size):
            original_feats = all_z[i, :num_augments//2]
            augmented_feats = all_z[i, num_augments//2:]
            
            # 通过自编码器
            orig_invariant, orig_variant, orig_recon, orig_bottleneck = self.autoencoder(original_feats)
            aug_invariant, aug_variant, aug_recon, aug_bottleneck = self.autoencoder(augmented_feats)
            
            # 更新内存库
            self.update_memory_bank(aug_bottleneck.detach())
            
            # 不变特征的InfoNCE损失（最大化互信息）
            invariant_loss += self.compute_info_nce_loss(orig_invariant, aug_invariant)
            
            # 可变特征的InfoNCE损失（最小化互信息）
            variant_loss += -self.compute_info_nce_loss(orig_variant, aug_variant)
            
            # 重建损失
            recon_loss = (self.reconstruction_criterion(orig_recon, original_feats) + 
                         self.reconstruction_criterion(aug_recon, augmented_feats)) / 2
            
            disentangle_loss += invariant_loss + variant_loss
            reconstruction_loss += recon_loss
            
        disentangle_loss /= batch_size
        reconstruction_loss /= batch_size
        
        total_loss = (contrastive_loss + 
                     self.hparams.get("disentangle_weight", 0.1) * disentangle_loss +
                     self.hparams.get("reconstruction_weight", 0.1) * reconstruction_loss)
        
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def predict(self, x, training=False):
        self.featurizer.train(training)
        features = self.featurizer(x)
        invariant, variant, _, _ = self.autoencoder(features)
        return torch.cat([invariant, variant], dim=1)

    def initialize_memory_bank(self, dataloader):
        self.memory_bank = torch.zeros(
            self.memory_bank_size, 
            self.hparams.get("bottleneck_dim", self.featurizer.n_outputs),
            device=next(self.parameters()).device
        )
        
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= self.memory_bank_size:
                    break
                
                x = x.to(self.memory_bank.device)
                features = self.featurizer(x)
                _, _, _, bottleneck = self.autoencoder(features)
                
                n = min(bottleneck.shape[0], self.memory_bank_size - i)
                self.memory_bank[i:i+n] = bottleneck[:n]

    def extract_sequence_features(self, sequences):
        """
        提取序列特征
        sequences: [N, frame_length, H, W, C]
        return: [N, feature_dim]
        """
        N, L = sequences.shape[:2]
        sequences = sequences.reshape(-1, *sequences.shape[2:])
        
        # 提取特征
        features = self.featurizer(sequences)
        features = features.reshape(N, L, -1)
        
        # 对时间维度进行池化
        features = torch.mean(features, dim=1)
        
        return features
