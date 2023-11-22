import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResdualBlock

class VAE_Ecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
                         #(batch_size, channel, row, col) -> (batch_size, 128, row, col)
                         nn.Conv2d(3, 128, kernel_size=3, padding=1),
                         
                         #(batch_size, 128, row, col) -> (batch_size, 128, row, col)
                         VAE_ResdualBlock(128, 128),
                         
                         #(batch_size, 128, row, col) -> (batch_size, 128, row, col)
                         VAE_ResdualBlock(128, 128),
                         
                         #(batch_size, 128, row, col) -> (batch_size, 128, row/2, col/2)
                         nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
                         
                         #(batch_size, 128, row/2, col/2) -> (batch_size, 256, row/2, col/2)
                         VAE_ResdualBlock(128, 256),
                         
                         #(batch_size, 256, row/2, col/2) -> (batch_size, 256, row/2, col/2)
                         VAE_ResdualBlock(256, 256),
                         
                         #(batch_size, 256, row/2, col/2) -> (batch_size, 256, row/4, col/4)
                         nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
                         
                         #(batch_size, 256, row/4, col/4) -> (batch_size, 512, row/4, col/4)
                         VAE_ResdualBlock(256, 512),
                         
                         #(batch_size, 512, row/4, col/4) -> (batch_size, 512, row/4, col/4)
                         VAE_ResdualBlock(512, 512),
                         
                         #(batch_size, 512, row/4, col/4) -> (batch_size, 512, row/8, col/8)
                         nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         VAE_ResdualBlock(512, 512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         VAE_ResdualBlock(512, 512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         VAE_ResdualBlock(512, 512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         VAE_AttentionBlock(512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         VAE_ResdualBlock(512, 512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         nn.GroupNorm(32, 512),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 512, row/8, col/8)
                         nn.SiLU(),
                         
                         #(batch_size, 512, row/8, col/8) -> (batch_size, 8, row/8, col/8)
                         nn.Conv2d(512, 8, kernel_size=3, padding=1),
                         
                         #(batch_size, 8, row/8, col/8) -> (batch_size, 8, row/8, col/8)
                         nn.Conv2d(8, 8, kernel_size=1, padding=0)
                         )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, row, col)
        # noise: (batch_size, out_channels, row/8, col/8)
        
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                #(padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        #split tensor (batch_size, 8, row/8, col/8) -> (batch_size, 4, row/8, col/8), (batch_size, 4, row/8, col/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        #(batch_size, 4, row/8, col/8) -> (batch_size, 4, row/8, col/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # x = mean + stdev*Z
        x = mean + (stdev*noise)
        
        #scaling
        x *= 0.18215
        
        return x
