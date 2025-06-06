import torch
import torch.nn as nn
from einops import rearrange
import math


class FRFN(nn.Module):
    def __init__(self, dim=30, hidden_dim=60, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4 #15
        self.dim_untouched = self.dim - self.dim_conv #45
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        B, C, H, W = x.shape
        x = x.reshape(B, -1, C).transpose(1, 2).contiguous().view(B, H * W, C)
        x_init = x

        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)
        x = x + x_init
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        return x


if __name__ == '__main__':
    # Instantiate the FRFN class
    dim = 64  # Dimension of input features


    # Create an instance of the FRFN module
    frfn = FRFN(dim)

    # Generate a random input tensor
    B = 1  # Batch size
    H = 64  # Height of the feature map
    W = 64  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, H * W, C)

    # Forward pass
    output = frfn(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())