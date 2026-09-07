import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind, c):
    if kind == "group":
        return nn.GroupNorm(min(8, c), c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unsupported norm: {kind}")


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018)."""

    def __init__(self, encoder_channels, decoder_channels, intermediate_channels, norm="group"):
        super().__init__()
        self.encoder_projection = nn.Sequential(
            nn.Conv2d(encoder_channels, intermediate_channels, 1, bias=False),
            _norm(norm, intermediate_channels),
        )
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, intermediate_channels, 1, bias=False),
            _norm(norm, intermediate_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, encoder_features, decoder_features):
        a = self.encoder_projection(encoder_features) + self.decoder_projection(decoder_features)
        return encoder_features * self.attention(a)


class UNetBase(nn.Module):
    """Shared trunk: 3-level U-Net, one bilinear step to out_size, global
    residual, refinement convs at full resolution. Subclasses decide what
    happens on the skip connections via _skip()."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        out_size=(2100, 2550),
        norm="group",
        kernel_size=7,
        refine_channels=32,
        width=1.0,
        coord_channels=False,
        clamp_output=False,
        reproject=False,
        reproject_first=False,
        reproject_mode="nearest",
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.out_size = out_size
        self.norm = norm
        self.k = kernel_size
        self.coord_channels = coord_channels
        self.clamp_output = clamp_output
        self.in_channels = in_channels

        # OSISAF -> MASAM2 projection, replacing the offline `cdo remapnn`.
        #   reproject=True        in the HEAD: the U-Net runs on the native
        #                         432x432 grid, only decoder features and the
        #                         residual base are warped.
        #   reproject_first=True  on the INPUT: the whole U-Net runs at
        #                         2100x2550. Needs use_checkpoint=True.
        if reproject and reproject_first:
            raise ValueError("choose one placement: reproject (head) or reproject_first (input)")
        self.reproject = reproject
        self.reproject_first = reproject_first
        if reproject or reproject_first:
            from reprojection import OsisafToMasam2, OUT_SHAPE
            self.resampler = OsisafToMasam2(mode=reproject_mode)
            self.out_size = OUT_SHAPE

        c_in = in_channels + (2 if coord_channels else 0)

        w = lambda c: max(8, int(round(c * width / 8)) * 8)   # keep GroupNorm groups happy
        c1, c2, c3, c4 = w(16), w(32), w(64), w(128)
        self.widths = (c1, c2, c3, c4)

        self.enc1 = self._block(c_in, c1)
        self.enc2 = self._block(c1, c2)
        self.enc3 = self._block(c2, c3)
        self.bottleneck = self._block(c3, c4)

        self.upconv1 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec1 = self._block(c3 * 2, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = self._block(c2 * 2, c2)
        self.upconv3 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec3 = self._block(c1 * 2, c1)

        r = refine_channels
        self.refine = nn.Sequential(
            nn.Conv2d(c1, r, 3, padding=1), _norm(norm, r), nn.ReLU(inplace=True),
            nn.Conv2d(r, r, 3, padding=1), _norm(norm, r), nn.ReLU(inplace=True),
            nn.Conv2d(r, out_channels, 3, padding=1),
        )
        # start as a pure pass-through of the bilinear branch
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def _skip(self, level, encoder_features, decoder_features):
        raise NotImplementedError

    def _block(self, ci, co):
        p = self.k // 2
        return nn.Sequential(
            nn.Conv2d(ci, co, self.k, padding=p), _norm(self.norm, co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, self.k, padding=p), _norm(self.norm, co), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return up

    def _run(self, block, x):
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        if self.reproject_first:
            x = self.resampler(x)

        src = x[:, : self.in_channels]           # physical field, before coords

        if self.coord_channels:
            b, _, h, w = x.shape
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            x = torch.cat([x, gy.expand(b, 1, h, w), gx.expand(b, 1, h, w)], dim=1)

        e1 = self._run(self.enc1, x)
        e2 = self._run(self.enc2, F.max_pool2d(e1, 2))
        e3 = self._run(self.enc3, F.max_pool2d(e2, 2))
        b_ = self._run(self.bottleneck, F.max_pool2d(e3, 2))

        # _match first: at 2100x2550 the decoder output does not land on the
        # encoder's shape (not divisible by 8)
        u1 = self._match(self.upconv1(b_), e3)
        d1 = self._run(self.dec1, torch.cat([u1, self._skip(0, e3, u1)], 1))
        u2 = self._match(self.upconv2(d1), e2)
        d2 = self._run(self.dec2, torch.cat([u2, self._skip(1, e2, u2)], 1))
        u3 = self._match(self.upconv3(d2), e1)
        d3 = self._run(self.dec3, torch.cat([u3, self._skip(2, e1, u3)], 1))
        del e1, e2, e3, b_, d1, d2, u1, u2, u3

        if self.reproject:
            up = self.resampler(d3)
            base = self.resampler(src)
        else:
            up = F.interpolate(d3, size=self.out_size, mode="bilinear", align_corners=False)
            base = F.interpolate(src, size=self.out_size, mode="bilinear", align_corners=False)

        out = base + self.refine(up)             # global residual

        return out.clamp(0.0, 1.0) if self.clamp_output else out


class UNetLight(UNetBase):
    """Plain skip connections."""

    def _skip(self, level, encoder_features, decoder_features):
        return encoder_features


class AttentionUNet(UNetBase):
    """Encoder skips gated by the decoder branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c1, c2, c3, _ = self.widths
        self.gates = nn.ModuleList([
            AttentionGate(c3, c3, max(8, c3 // 2), norm=self.norm),
            AttentionGate(c2, c2, max(8, c2 // 2), norm=self.norm),
            AttentionGate(c1, c1, max(8, c1 // 2), norm=self.norm),
        ])

    def _skip(self, level, encoder_features, decoder_features):
        return self.gates[level](encoder_features, decoder_features)
