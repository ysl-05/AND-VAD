from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Base module for processing timestep embeddings.
    Any module that needs to process timestep embeddings should inherit from this class.
    """

    def __init__(self, channels, emb_channels, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        
        # Timestep embedding processing layers
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * channels,  # for scale and shift
            ),
        )
        
        # Main processing layers
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(3, channels, channels, 3, padding=1),
        )
        
        self.out_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(3, channels, channels, 3, padding=1)),
        )

    def forward(self, x, emb):
        """
        Apply module to input tensor while processing timestep embedding.

        :param x: Input tensor [B, C, D, H, W]
        :param emb: Timestep embedding [B, emb_channels]
        :return: Processed tensor [B, C, D, H, W]
        """
        # Process timestep embedding
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
            
        # Split scale and shift
        scale, shift = th.chunk(emb_out, 2, dim=1)
        
        # Apply main processing layers
        h = self.in_layers(x)
        
        # Apply scale and shift
        h = h * (1 + scale) + shift
        
        # Apply output layers
        h = self.out_layers(h)
        
        return h


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, seq_factor, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.seq_factor = seq_factor
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.seq_factor:
                d = self.seq_factor
            else:
                d = x.shape[2]
            # print('upsample d', d)
            x = F.interpolate(
                x, (d, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, seq_factor, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if seq_factor == 0:
            d = 1
        else:
            d = 2
        print('downsample d', d)
        stride = 2 if dims != 3 else (d, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__(channels, emb_channels, dropout)
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class PositionalEmbedding(nn.Module):
    
    def __init__(self, num_channels, max_positions=10000, state_dim=6):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.state_dim = state_dim
        
        # 
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels)
        )
        
        # 
        position = th.arange(max_positions).unsqueeze(1)
        div_term = th.exp(th.arange(0, num_channels, 2) * (-math.log(10000.0) / num_channels))
        pe = th.zeros(max_positions, num_channels)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, state_info=None):
        
        B, C, D, H, W = x.shape
        
        # 处理状态信息
        if state_info is not None:
            # 将状态信息转换为嵌入
            state_emb = self.state_embedding(state_info)  # [B, D, C]
            state_emb = state_emb.unsqueeze(-1).unsqueeze(-1)  # [B, D, C, 1, 1]
            state_emb = state_emb.expand(-1, -1, -1, H, W)  # [B, D, C, H, W]
            state_emb = state_emb.permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]
        else:
            state_emb = th.zeros(B, C, D, H, W, device=x.device)
        
        # 获取每个维度的位置编码
        d_pos = self.pe[:D].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 1, D, 1, 1, C]
        h_pos = self.pe[:H].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)    # [1, 1, 1, H, 1, C]
        w_pos = self.pe[:W].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)     # [1, 1, 1, 1, W, C]
        
        # 扩展位置编码到batch size
        d_pos = d_pos.expand(B, -1, -1, H, W, -1)
        h_pos = h_pos.expand(B, -1, D, -1, W, -1)
        w_pos = w_pos.expand(B, -1, D, H, -1, -1)
        
        # 合并三个维度的位置编码
        pos_embedding = d_pos + h_pos + w_pos
        pos_embedding = pos_embedding.permute(0, 5, 1, 2, 3, 4).squeeze(-1).squeeze(-1)
        
        # 将状态编码和位置编码结合
        return pos_embedding + state_emb


class UNetModel(nn.Module):
    """
    3D U-Net model with timestep embedding and attention mechanism.
    Supports injecting conditional information from pre-trained 3D encoder at different resolutions.
    
    :param in_channels: Number of input channels
    :param model_channels: Base channel count
    :param out_channels: Number of output channels
    :param num_res_blocks: Number of residual blocks per downsample level
    :param attention_resolutions: Layers to use attention
    :param scale_time_dim: Time dimension scaling factor
    :param dropout: Dropout rate
    :param channel_mult: Channel multiplier for each level
    :param conv_resample: Whether to use learned convolutions for up/downsampling
    :param dims: Number of dimensions (3 for 3D)
    :param num_classes: Number of classes (for conditional generation)
    :param use_checkpoint: Whether to use gradient checkpointing
    :param num_heads: Number of attention heads
    :param num_heads_upsample: Number of attention heads for upsampling
    :param use_scale_shift_norm: Whether to use scale-shift normalization
    :param use_positional_embedding: Whether to use positional embedding
    :param state_dim: Dimension of state information
    :param z_ir_channels: Number of channels for low-resolution condition (z_ir)
    :param z_sp_channels: Number of channels for high-resolution condition (z_sp)
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        scale_time_dim,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_positional_embedding=True,
        state_dim=6,
        z_ir_channels=64,
        z_sp_channels=256,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.time_reduction = [scale_time_dim]
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.z_ir_channels = z_ir_channels
        self.z_sp_channels = z_sp_channels

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Positional embedding
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_embedding = PositionalEmbedding(model_channels, state_dim=state_dim)
            self.pos_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        # Class conditioning
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input layer with z_ir conditioning
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + z_ir_channels, model_channels, 3, padding=1)
                )
            ]
        )
        
        # Downsampling path
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                seq_factor = self.time_reduction[-1] // 2
                self.time_reduction.append(seq_factor)
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, seq_factor=seq_factor)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Upsampling path with z_sp conditioning
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, seq_factor=self.time_reduction[level - 1])
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output layer
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, state_info=None, z_ir=None, z_sp=None, return_features=False):
        """
        Forward pass with conditional information from pre-trained 3D encoder
        
        :param x: Input tensor [B, C, D, H, W]
        :param timesteps: Timesteps [B]
        :param y: Class labels [B]
        :param state_info: State information [B, D, state_dim]
        :param z_ir: Low-resolution condition from encoder [B, z_ir_channels, D, H, W]
        :param z_sp: High-resolution condition from encoder [B, z_sp_channels, D, H, W]
        :param return_features: Whether to return intermediate features
        :return: Output tensor, if return_features=True also returns intermediate features list
        """
        assert (y is not None) == (self.num_classes is not None)
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Positional embedding
        if self.use_positional_embedding:
            pos_emb = self.pos_embedding(x, state_info)
            pos_emb = self.pos_embed(pos_emb.view(pos_emb.shape[0], -1))
            emb = emb + pos_emb

        # Class conditioning
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # Store features
        hs = []
        features = [] if return_features else None
        
        # Concatenate z_ir with input
        if z_ir is not None:
            # Reshape z_ir to match input dimensions if needed
            if z_ir.shape[2:] != x.shape[2:]:
                z_ir = F.interpolate(z_ir, size=x.shape[2:], mode='trilinear', align_corners=False)
            h = th.cat([x, z_ir], dim=1)
        else:
            h = x
            
        h = h.type(self.inner_dtype)
        
        # Downsampling path
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            if return_features:
                features.append(h)
        
        # Middle block
        h = self.middle_block(h, emb)
        if return_features:
            features.append(h)
        
        # Upsampling path with z_sp conditioning
        for i, module in enumerate(self.output_blocks):
            cat_in = th.cat([h, hs.pop()], dim=1)
            
            # Inject z_sp at appropriate resolution
            if z_sp is not None and i < len(self.output_blocks) - 1:
                current_size = cat_in.shape[2:]
                if z_sp.shape[2:] != current_size:
                    z_sp_resized = F.interpolate(z_sp, size=current_size, mode='trilinear', align_corners=False)
                else:
                    z_sp_resized = z_sp
                cat_in = th.cat([cat_in, z_sp_resized], dim=1)
                
            h = module(cat_in, emb)
            if return_features:
                features.append(h)
        
        # Output
        output = self.out(h)
        
        if return_features:
            return output, features
        return output

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

