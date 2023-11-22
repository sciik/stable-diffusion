import torch
from torch import nn
from torch.nn import funcional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, 4*n_embed)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        
        # x: (1, 1280)
        return x


class UnetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

def forward(self, feature, time):
    # feature: (batch_size, inchannels, row, col)
    # time (1, 1280)
    
    residual = feature
        
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        merged = merged + self.residual_layer(residual)
        
        return merged


class UnetAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (batch_size, features, row, col)
        # context: (batch_size, seq_len, dim)
        
        residual_long = x
        
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        #  (batch_size, features, row, col) -> (batch_size, features, row*col)
        x = x.view((n, c, h*w))
        
        #  (batch_size, features, row*col) -> (batch_size, row*col, features)
        x = x.transpose(-1, -2)
        
        # normalization + self attention with skip connection
        residual_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residual_short(x)
        
        # normalization + cross attention with skip connection
        residual_short = x
        
        x = self.layernorm_2(x)
        self.attention_2(x, context)
        x += residual_short(x)
        
        # normalization + FF with GeGLU and skip connection
        residual_short = x
        
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residual_short
        
        # (batch_size, row*col, features) -> (batch_size, features, row*col)
        x = x.transpose(-1, -2)
        
        # (batch_size, features, row*col) -> (batch_size, features, row, col)
        x = x.view((n, c, h, w))
        x = self.conv_output(x) + residual_long
        
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch_size,features, row, col) -> (batch_size,features, row*2, col*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UnetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UnetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

    return x


class Unet(nn.Module):
    def __init__(self):
        super.__init__()
        self.encoders = nn.Module([
                                   # (batch_size, 4, row/8, col/8) -> (batch_size, 320, row/8, col/8)
                                   SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                                   SwitchSequential(UnetResidualBlock(320, 320, UnetAttentionBlock(8, 40))),
                                   SwitchSequential(UnetResidualBlock(320, 320, UnetAttentionBlock(8, 40))),
                                   
                                   # (batch_size, 320, row/8, col/8) -> (batch_size, 640, row/16, col/16)
                                   SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
                                   SwitchSequential(UnetResidualBlock(320, 640, UnetAttentionBlock(8, 80))),
                                   SwitchSequential(UnetResidualBlock(640, 640, UnetAttentionBlock(8, 80))),
                                   
                                   # (batch_size, 640, row/16, col/16) -> (batch_size, 1280, row/32, col/32)
                                   SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
                                   SwitchSequential(UnetResidualBlock(640, 1280, UnetAttentionBlock(8, 160))),
                                   SwitchSequential(UnetResidualBlock(1280, 1280, UnetAttentionBlock(8, 160))),
                                   
                                   # (batch_size, 1280, row/32, col/32) -> (batch_size, 1280, row/64, col/64)
                                   SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
                                   SwitchSequential(UnetResidualBlock(1280, 1280)),
                                   SwitchSequential(UnetResidualBlock(1280, 1280))
                                   ])
            
                                   self.bottelnet = SwitchSequential(
                                                                     UnetResidualBlock(1280, 1280),
                                                                     UnetAttentionBlock(8, 160),
                                                                     UnetResidualBlock(1280, 1280)
                                                                     )
                                   
                                   self.decoders = nn.ModuleList([
                                                                  # (batch_size, 2560, row/64, col/64) ->(batch_size, 1280, row/64, col/64)
                                                                  SwitchSequential(UnetResidualBlock(2560, 1280)),
                                                                  SwitchSequential(UnetResidualBlock(2560, 1280)),
                                                                  
                                                                  SwitchSequential(UnetResidualBlock(2560, 1280), Upsample(1280)),
                                                                  SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),
                                                                  SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),
                                                                  
                                                                  SwitchSequential(UnetResidualBlock(1920, 1280), UnetAttentionBlock(8, 160), Upsample(1280)),
                                                                  SwitchSequential(UnetResidualBlock(1920, 640), UnetAttentionBlock(8, 80)),
                                                                  SwitchSequential(UnetResidualBlock(1280, 640), UnetAttentionBlock(8, 80)),
                                                                  
                                                                  SwitchSequential(UnetResidualBlock(960, 640), UnetAttentionBlock(8, 160), Upsample(640)),
                                                                  SwitchSequential(UnetResidualBlock(960, 320), UnetAttentionBlock(8, 40)),
                                                                  SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 80)),
                                                                  SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 40)),
                                                                  ])


class UnetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (batch_size, 320, row/8, col/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        
        # (batch_size, 4, row/8, col/8)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = UnetOutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent: (batch_size, 4, row/8, col/8)
        #context: (batch_size, seq_len, dim)
        #time: (1, 320)
        
        #(1, 320) ->(1, 1280)
        time = self.time_embedding(time)
        
        # (batch, 4, row/8, col/8) -> (batch, 320, row/8, col/8)
        output = self.unet(latent, context, time)
        
        # (batch, 320, row/8, col/8) -> (batch, 4, row/8, col/8)
        output = self.final(output)
        
        return output
