import copy
from collections import OrderedDict
from xmlrpc.client import Fault

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from ...core import register
from .utils import get_activation
from .defe import LiteDeFE
from .umap_conv_atttention import ConvolutionalAttention, downsample_prior
__all__ = ["HybridEncoder"]

class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class ELAN(nn.Module):

    def __init__(self, c1, c2, c3, c4, n=2, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class RepNCSPELAN4(nn.Module):

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden_channels, hidden_channels, act=get_activation(act))
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class RelationAwareAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        num_relation_heads=None,
        bias=True,
        tau=0.3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = int(num_heads)
        self.num_relation_heads = self.num_heads if num_relation_heads is None else int(num_relation_heads)
        assert 0 <= self.num_relation_heads <= self.num_heads, (
            f"num_relation_heads must be in [0, {self.num_heads}], got {self.num_relation_heads}"
        )

        self.mha = nn.MultiheadAttention(
            embed_dim,
            self.num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

        self.tau = tau

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        rename_map = {
            "in_proj_weight": "mha.in_proj_weight",
            "in_proj_bias": "mha.in_proj_bias",
            "out_proj.weight": "mha.out_proj.weight",
            "out_proj.bias": "mha.out_proj.bias",
        }

        for old_suffix, new_suffix in rename_map.items():
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        prior=None,

        relation_weights=None,
        need_weights=False,
        average_attn_weights=True,
        skip_relation=False,
    ):
        merged_mask = self._merge_attn_mask(
            query=query,
            key=key,
            attn_mask=attn_mask,
            prior=prior,
            relation_weights=relation_weights,
            skip_relation=skip_relation,
        )
        return self.mha(
            query,
            key,
            value,
            attn_mask=merged_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

    @staticmethod
    def _to_additive_bias(mask, dtype):

        if mask.dtype == torch.bool:
            bias = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
            return bias.masked_fill(mask, float("-inf"))
        return mask.to(dtype=dtype)

    @staticmethod
    def _infer_hw_from_L_and_prior(L, prior_hw=None):

        if L <= 0:
            raise ValueError(f"L must be > 0, got {L}")

        factors = []
        root = int(math.sqrt(L))
        for h in range(1, root + 1):
            if L % h == 0:
                w = L // h
                factors.append((h, w))
                factors.append((w, h))

        if not factors:
            raise ValueError(f"Failed to factorize L={L}")

        if prior_hw is None:

            return min(factors, key=lambda t: abs(t[0] - t[1]))

        ph, pw = prior_hw
        target_ratio = pw / max(ph, 1)

        def score(pair):
            h, w = pair
            return abs((w / max(h, 1)) - target_ratio)

        return min(factors, key=score)

    def _prior_to_u(self, prior, L, device, dtype):

        if prior.dim() == 4:

            prior_ = prior.mean(dim=1, keepdim=True) if prior.shape[1] != 1 else prior
            Hp, Wp = int(prior_.shape[-2]), int(prior_.shape[-1])
            h, w = self._infer_hw_from_L_and_prior(L, prior_hw=(Hp, Wp))

            p = downsample_prior(prior_.to(device=device, dtype=dtype), h, w)
            u = p.flatten(2).squeeze(1)
        elif prior.dim() == 3:

            if prior.shape[1] == 1 and prior.shape[2] == L:
                u = prior[:, 0, :]
            elif prior.shape[2] == 1 and prior.shape[1] == L:
                u = prior[:, :, 0]
            elif prior.shape[1] == L:
                u = prior.mean(dim=-1)
            else:
                raise ValueError(f"Unsupported prior shape {tuple(prior.shape)} for L={L}")
            u = u.to(device=device, dtype=dtype)
        elif prior.dim() == 2:
            if prior.shape[1] != L:
                raise ValueError(f"prior shape {tuple(prior.shape)} does not match L={L}")
            u = prior.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"prior must be 2D/3D/4D, got {prior.dim()}D")

        return u.clamp(0.0, 1.0)

    def _build_log_relation_from_prior(self, prior, L, device, dtype):

        u = self._prior_to_u(prior, L=L, device=device, dtype=dtype)

        u_i = u.unsqueeze(1).unsqueeze(-1)
        u_j = u.unsqueeze(1).unsqueeze(-2)

        tau = self.tau
        relation_weights = torch.exp(-(u_i - u_j).abs() / tau) + 1e-6
        return relation_weights.log()

    def _merge_attn_mask(self, query, key, attn_mask, prior, relation_weights, skip_relation):

        use_relation = (
            (not skip_relation)
            and (self.num_relation_heads > 0)
            and ((relation_weights is not None) or (prior is not None))
        )

        if (not use_relation) and (attn_mask is None):
            return None

        B, Lq, _ = query.shape
        _, Lk, _ = key.shape
        dtype = query.dtype
        device = query.device

        base_bias = None
        if attn_mask is not None:
            base_bias = self._to_additive_bias(attn_mask, dtype=dtype)

            if base_bias.dim() == 2:
                pass
            elif base_bias.dim() == 3:
                if base_bias.shape[0] not in (B, B * self.num_heads):
                    raise ValueError(
                        f"attn_mask 3D first dim must be B or B*num_heads; got {base_bias.shape[0]}, B={B}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or 3D; got {base_bias.dim()}D")

        if not use_relation:
            return attn_mask

        Hr = self.num_relation_heads
        H = self.num_heads

        if relation_weights is not None:
            rel = relation_weights
            if rel.dim() == 3:
                rel = rel[:, None, :, :]
            if rel.dim() != 4:
                raise ValueError(
                    f"relation_weights must be (B,Lq,Lk) or (B,1/Hr/H,Lq,Lk); got {tuple(rel.shape)}"
                )
            if rel.shape[0] != B or rel.shape[-2:] != (Lq, Lk):
                raise ValueError(
                    f"relation_weights shape must match (B,*,Lq,Lk)=({B},*,{Lq},{Lk}); got {tuple(rel.shape)}"
                )

            rel = rel.clamp(min=1e-6).log().to(dtype=dtype, device=device)

            if rel.shape[1] == 1:
                rel_log = rel.expand(B, Hr, Lq, Lk)
            elif rel.shape[1] == Hr:
                rel_log = rel
            elif rel.shape[1] == H:
                rel_log = rel[:, -Hr:, :, :]
            else:
                raise ValueError(
                    f"relation_weights head dim must be 1, Hr={Hr}, or H={H}; got {rel.shape[1]}"
                )
        else:

            if Lq != Lk:
                raise ValueError(f"prior-based relation assumes self-attn (Lq==Lk), got {Lq} vs {Lk}")
            log_rel = self._build_log_relation_from_prior(prior, L=Lq, device=device, dtype=dtype)
            rel_log = log_rel.expand(B, Hr, Lq, Lk)

        rel_bias = torch.zeros((B, H, Lq, Lk), device=device, dtype=dtype)
        rel_bias[:, -Hr:, :, :] = rel_log

        merged = rel_bias
        if base_bias is not None:
            if base_bias.dim() == 2:
                merged = merged + base_bias[None, None, :, :]
            else:
                if base_bias.shape[0] == B:
                    merged = merged + base_bias[:, None, :, :]
                else:
                    merged = merged + base_bias.reshape(B, H, Lq, Lk)

        return merged.reshape(B * H, Lq, Lk)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_relation_heads=None,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = RelationAwareAttention(d_model, nhead, dropout=dropout, batch_first=True, num_relation_heads=num_relation_heads)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, prior=None, relation_weights=None, skip_relation=False) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, prior=prior, relation_weights=relation_weights, skip_relation=skip_relation)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, prior=None, relation_weights=None, skip_relation=False) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, prior=prior, relation_weights=relation_weights, skip_relation=skip_relation)

        if self.norm is not None:
            output = self.norm(output)

        return output

@register()
class HybridEncoder(nn.Module):
    __share__ = [
        "eval_spatial_size",
    ]

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_defe: bool = True,
        defe_type: str = "light",
        use_umap_attn: bool = True,
        umap_pdim: int = 128,
        umap_sk_size = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )

            self.input_proj.append(proj)

        self.use_defe = use_defe
        self.defe_type = defe_type
        if self.use_defe:
            if self.defe_type != "light":
                raise ValueError(f"Invalid defe_type: {self.defe_type}. Currently supported: 'light'")
            self.DeFE = LiteDeFE()

        self.use_umap_attn = use_umap_attn
        if self.use_umap_attn:

            self.umap_attn = ConvolutionalAttention(
                pdim=umap_pdim,
                proj_dim_in=64,
                use_weighted_pool=True,
                concat_prior_scalar=True,
            )

            self.umap_lk = nn.Conv2d(
                umap_pdim, umap_pdim,
                kernel_size=umap_sk_size,
                padding=umap_sk_size // 2,
                bias=False
            )

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            num_relation_heads=4,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )

            )

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    SCDown(hidden_dim, hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )

            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        out = {}
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        prior = None

        if self.use_defe:
            defe_feature = self.DeFE(proj_feats[0])
            out["defe"] = {"density_map": defe_feature}

            if self.use_umap_attn:
                prior = defe_feature
                x = proj_feats[0]
                proj_feats[0] = self.umap_attn(x, prior=prior, lk_filter=self.umap_lk.weight)

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]

                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                attn_prior = prior.detach() if (prior is not None and enc_ind == max(self.use_encoder_idx)) else None
                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed, prior=attn_prior, skip_relation=(attn_prior is None))
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                )

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            pan_out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_height], dim=1),
            )
            outs.append(pan_out)

        out["feats"] = outs
        return out
