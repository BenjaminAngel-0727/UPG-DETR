import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from .utils import get_activation

import torch
import torch.nn.functional as F

def downsample_prior(prior: torch.Tensor, h: int, w: int) -> torch.Tensor:

    if prior is None:
        return None
    if prior.dim() != 4 or prior.size(1) != 1:
        raise ValueError(f"prior must be [B,1,H,W], got {tuple(prior.shape)}")
    Hp, Wp = prior.size(-2), prior.size(-1)

    if (Hp, Wp) == (h, w):
        return prior.clamp(0.0, 1.0)

    if (Hp % h != 0) or (Wp % w != 0):
        raise ValueError(
            f"prior size ({Hp},{Wp}) is not an integer multiple of feature size ({h},{w})."
        )
    sh, sw = Hp // h, Wp // w
    if sh != sw:
        raise ValueError(
            f"prior scale factors differ on H/W: sh={sh}, sw={sw} for "
            f"prior ({Hp},{Wp}) -> feat ({h},{w})."
        )

    p = F.avg_pool2d(prior, kernel_size=sh, stride=sh)
    return p.clamp(0.0, 1.0)

def weighted_gap(feat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    w = weight
    num = (feat * w).sum(dim=(2, 3), keepdim=True)
    den = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return num / den

class ConvolutionalAttention(nn.Module):
    def __init__(
        self,
        pdim: int,
        proj_dim_in: Optional[int] = None,
        use_weighted_pool: bool = True,
        concat_prior_scalar: bool = False,
    ):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = 3

        self.use_weighted_pool = use_weighted_pool
        self.concat_prior_scalar = concat_prior_scalar

        kgen_in_ch = self.proj_dim_in + (1 if concat_prior_scalar else 0)

        self.kgen = nn.Sequential(
            nn.Conv2d(kgen_in_ch, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0),
        )
        nn.init.zeros_(self.kgen[-1].weight)
        nn.init.zeros_(self.kgen[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:

        b, c, h, w = x.shape
        p = downsample_prior(prior, h, w)

        x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

        proj_feat = x[:, :self.proj_dim_in]

        if (p is not None) and self.use_weighted_pool:
            pooled = weighted_gap(proj_feat, p)
        else:
            pooled = F.adaptive_avg_pool2d(proj_feat, 1)

        if (p is not None) and self.concat_prior_scalar:
            p_scalar = F.adaptive_avg_pool2d(p, 1)
            pooled = torch.cat([pooled, p_scalar], dim=1)

        dynamic_kernel = self.kgen(pooled).reshape(-1, 1, self.sk_size, self.sk_size)

        x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
        x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=b * self.pdim)
        x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=b, c=self.pdim)

        x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_

        return torch.cat([x1, x2], dim=1)

    def extra_repr(self):
        return (f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}, '
                f'use_weighted_pool={self.use_weighted_pool}, concat_prior_scalar={self.concat_prior_scalar}')
