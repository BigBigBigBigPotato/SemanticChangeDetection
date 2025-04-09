from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM_AD(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d',
                 interaction_cfg=(None,None,None,None), **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

        # cross-correlation
        self.css=[]
        for ia_layer in interaction_cfg:
            if ia_layer is None:
                ia_layer = Indentity()
            self.css.append(ia_layer)
        self.css=nn.ModuleList(self.css)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x1, x2):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        outs1, outs2 = [], []
        for i, layer in enumerate(self.layers):
            o1, x1 = layer_forward(layer, x1)  # (B, H, W, C)
            o2, x2 = layer_forward(layer, x2)
            x1, x2 = self.css[i](x1, x2)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out1 = norm_layer(o1)
                out2 = norm_layer(o2)
                if not self.channel_first:
                    out1 = out1.permute(0, 3, 1, 2).contiguous()
                    out2 = out2.permute(0, 3, 1, 2).contiguous()
                outs1.append(out1)
                outs2.append(out2)

        if len(self.out_indices) == 0:
            return x1, x2

        return outs1, outs2


class ChannelExchange(nn.Module):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class Aggregation_distribution(nn.Module):
    # Aggregation_Distribution Layer (AD)
    def __init__(self,
                 channels,
                 num_paths=2,
                 attn_channels=None,
                 act_cfg=nn.ReLU,
                 norm_cfg=nn.BatchNorm2d):
        """聚合分布层初始化

        Args:
            channels: 输入通道数
            num_paths: 路径数量，默认为2
            attn_channels: 注意力通道数，如果为None则为channels//16
            act_cfg: 激活函数配置
            norm_cfg: 标准化层配置
        """
        super().__init__()
        self.num_paths = num_paths  # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_cfg(attn_channels)
        self.act = act_cfg()
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2

class Indentity(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()

    def forward(self,x1,x2):
        return x1,x2
