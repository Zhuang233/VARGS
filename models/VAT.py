from .build import MODELS
import torch.nn as nn
import torch
from math import pi

def fourier_encode(x, max_freqs, num_bands=4, concat_input=True):
    """
    Fourier encode the input with independent frequency ranges for each dimension.

    Args:
        x: Input tensor of shape [batch_size, n, d], where `n` is the number of points
           and `d` is the number of dimensions.
        max_freqs: List or tensor of length `d`, specifying the maximum frequency for each dimension.
        num_bands: Number of frequency bands to use per dimension.
        concat_input: Whether to concatenate the original input to the Fourier features.

    Returns:
        Tensor of shape [batch_size, n, d * (2 * num_bands) + (d if concat_input is True else 0)].
    """
    assert x.ndim == 3, "Input tensor x must have shape [batch_size, n, d]."
    assert len(max_freqs) == x.shape[2], "max_freqs must have the same length as the number of dimensions in x."

    # Extract batch size and input shape
    batch_size, n, d = x.shape

    # Prepare frequency scales for each dimension
    device, dtype = x.device, x.dtype
    max_freqs = torch.tensor(max_freqs, device=device, dtype=dtype)
    scales = torch.stack([torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype) for max_freq in max_freqs], dim=0)
    scales = scales.view(1, 1, d, num_bands)  # Shape: [1, 1, d, num_bands]

    # Expand input for frequency encoding
    x = x.unsqueeze(-1)  # Shape: [batch_size, n, d, 1]

    # Compute Fourier features
    x = x * scales * pi  # Shape: [batch_size, n, d, num_bands]
    fourier_features = torch.cat([x.sin(), x.cos()], dim=-1)  # Shape: [batch_size, n, d, 2 * num_bands]

    # Flatten features across dimensions
    fourier_features = fourier_features.view(batch_size, n, -1)  # Shape: [batch_size, n, d * (2 * num_bands)]

    # Optionally concatenate original input
    if concat_input:
        fourier_features = torch.cat((x.squeeze(-1), fourier_features), dim=-1)  # Shape: [batch_size, n, d + d * (2 * num_bands)]

    return fourier_features


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, num_queries):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))
        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8, batch_first=True)

    def forward(self, features):
        # Query features through cross-attention
        queries = self.queries.unsqueeze(0).expand(features.size(0), -1, -1)  # Batch the queries
        attended, _ = self.attention(queries, features, features)
        return attended

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=input_dim, nhead=8) for _ in range(num_layers)])

    def forward(self, features):
        # Pass through self-attention layers
        for layer in self.layers:
            features = layer(features)

        # Keep only the tokens
        return features[:, :self.tokens.size(0), :]

def sample_from_gaussian(features, latent_dim=8):
    """
    从给定的均值和方差中采样出高斯隐向量
    
    Args:
        vvq_features: 输入特征，大小为 [batch_size, num_tokens, 16]
        latent_dim: 隐空间的维度，默认为 8
    
    Returns:
        sampled_latents: 从高斯分布中采样的隐向量，大小为 [batch_size, num_tokens, latent_dim]
    """
    # 提取均值和方差
    means = features[:, :, :latent_dim]  # 前 8 个维度为均值
    log_var = features[:, :, latent_dim:]  # 后 8 个维度为对数方差

    # 对数方差转换为标准方差
    std_dev = torch.exp(0.5 * log_var)  # 因为方差是 log_var 的指数，因此 std_dev = exp(0.5 * log_var)
    
    # 从标准正态分布中采样
    epsilon = torch.randn_like(means)  # 生成与均值相同形状的标准正态分布噪声

    # 计算隐向量
    sampled_latents = means + std_dev * epsilon  # 使用 reparameterization trick 采样

    return sampled_latents


@MODELS.register_module()
class VAT(nn.Module):
    def __init__(self, config):
        super(VAT, self).__init__()
        self.config = config
        self.linear = nn.Linear(in_features=14*64+14, out_features=768)
        self.cross_attention = CrossAttentionLayer(query_dim = 768, num_queries = 3072)
        self.tokens = nn.Parameter(torch.randn(1024, 768))
        self.self_attention = SelfAttention(input_dim = 768, num_layers = 12)
        self.linear1 = nn.Linear(in_features=768, out_features=16)

        self.linear2 = nn.Linear(in_features=8, out_features=768)
        self.cross_attention2 = CrossAttentionLayer(query_dim = 768, num_queries = 3072)
        self.self_attention2 = SelfAttention(input_dim = 768, num_layers = 12)

        # Mipmap convolutional layers (Upsampling)
        self.conv_r = nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1)  # From 32x32 to 64x64
        self.conv_r2 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)  # From 64x64 to 128x128
        self.conv_r4 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)  # From 128x128 to 256x256



    def forward(self, x):
        # x shape: [batch_size, npoints, 14]
        max_freqs = [
            15.0, 15.0, 15.0,  # pos
            10.0,              # opacity
            10.0, 10.0, 10.0,  # scale
            20.0, 20.0, 20.0, 20.0,  # rotation
            8.0, 8.0, 8.0      # sh
        ]

        # 傅里叶编码，拓展特征
        x = fourier_encode(x, max_freqs, num_bands=64, concat_input=True)
        # x shape: [batch_size, npoints, 14*64+14]
        x = self.linear(x)
        # x shape: [batch_size, npoints, 768]
        x = self.cross_attention(x)
        # x shape: [batch_size, 3072, 768]

        # Concatenate tokens with features
        tokens = self.tokens.unsqueeze(0).expand(x.size(0), -1, -1) # ???? 可学习的token不能这样搞吧
        x = torch.cat([tokens, x], dim=1)

        x = self.self_attention(x)
        # x shape: [batch_size, 1024, 768]
        x= self.linear1(x)
        # x shape: [batch_size, 1024, 16]

        # 8+8 均值+方差
        sampled_latents = sample_from_gaussian(x, latent_dim=8)
        # sampled_latents shape: [batch_size, 1024, 8]

        # ToDo: VVQ 





        x = self.linear2(sampled_latents)
        # x shape: [batch_size, 1024, 768]
        x = self.cross_attention2(x)
        # x shape: [batch_size, 3072, 768]
        x = self.cross_attention2(x)
        # x shape: [batch_size, 3*32*32, 768]
        Triplane_feature = x.view(-1, 3, 32, 32, 768)


        # Create mipmaps by upsampling the triplane features
        mip_r = self.conv_r(Triplane_feature)  # From 32x32 to 64x64
        mip_r2 = self.conv_r2(mip_r)  # From 64x64 to 128x128
        mip_r4 = self.conv_r4(mip_r2)  # From 128x128 to 256x256

        # TODO: mipmaps to GS







        loss_dict = {}
        x = x.reshape(-1)
        out = self.model(x)
        mse_loss = nn.MSELoss()
        loss1 = mse_loss(out, x)
        loss_dict["loss1"] = loss1
        return loss_dict