import torch
import torch.nn.functional as F

from .utils.utils import bilinear_sampler

try:
    import alt_cuda_corr
except ImportError:
    # alt_cuda_corr is not compiled
    pass

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn(
            "failed to load custom correlation module"
            " which is needed for embedding visualizations",
            ImportWarning,
        )


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, compute_spatial=False):
        self.num_levels = num_levels
        self.radius = radius
        self.compute_spatial = compute_spatial
        self.corr_pyramid = []

        if self.compute_spatial:
            self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
            out_corr = spatial_correlation_sample(
                fmap1,
                fmap2,
                kernel_size=1,
                patch_size=21,
                stride=1,
                padding=0,
                dilation_patch=2,
            )
            # collate dimensions 1 and 2 in order to be treated as a
            # regular 4D tensor
            batch, ph, pw, h, w = out_corr.size()
            self.spatial_corr = out_corr.view(batch, ph * pw, h, w) / fmap1.size(
                1
            )  # for visualization purposes
            corr = out_corr.view(batch * ph * pw, 1, h, w)
            self.corr_pyramid.append(corr)
            for _ in range(self.num_levels - 1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                self.corr_pyramid.append(corr)
        else:
            # all pairs correlation
            corr = CorrBlock.corr(fmap1, fmap2)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

            self.corr_pyramid.append(corr)
            for _ in range(self.num_levels - 1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                self.corr_pyramid.append(corr)

    def get_corr_pyramid(self):
        return self.corr_pyramid

    def get_spatial_corr(self):
        return self.spatial_corr if self.compute_spatial else None

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = corr if self.compute_spatial else bilinear_sampler(corr, coords_lvl)
            if self.compute_spatial:
                for _ in range(i):
                    corr = self.upsample(corr)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for _ in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
