import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import AlternateCorrBlock, CorrBlock
from .extractor import BasicEncoder, FlowNetCEncoder, SmallEncoder
from .update import BasicUpdateBlock, SmallUpdateBlock
from .utils.utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args, return_feat_maps: bool = False):
        super().__init__()
        self.args = args
        self.return_feat_maps = return_feat_maps

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            # args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            # args.corr_levels = 4
            args.corr_radius = 4

        if "compute_spatial" not in self.args:
            self.args.compute_spatial = False
        if self.args.compute_spatial:
            self.args.compute_spatial = True

        if "dropout" not in self.args:
            self.args.dropout = 0

        if "alternate_corr" not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.flowNetCEnc and args.no_separate_context:
            self.fnet = FlowNetCEncoder(
                output_dim=256, norm_fn="none", dropout=args.dropout
            )
            self.conv_redir = nn.Conv2d(
                256, hdim + cdim, kernel_size=1, stride=1, padding=(1 - 1) // 2, bias=True
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        elif args.flowNetCEnc:
            self.fnet = FlowNetCEncoder(
                output_dim=256, norm_fn="none", dropout=args.dropout
            )
            self.cnet = FlowNetCEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        elif args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(
                output_dim=256,
                norm_fn=args.fnorm,
                dropout=args.dropout,
            )
            if args.no_separate_context:
                self.conv_redir = nn.Conv2d(
                    256, hdim + cdim, kernel_size=1
                )  # similar to conv_redir in FlowNetC
            else:
                self.cnet = BasicEncoder(
                    output_dim=hdim + cdim,
                    norm_fn=args.cnorm,
                    dropout=args.dropout,
                )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):  # pylint: disable=no-self-use
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):  # pylint: disable=no-self-use
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """Estimate optical flow between pair of frames"""
        iters = self.args.iters
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        if self.return_feat_maps:
            return_feat_maps = []

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.return_feat_maps:
            return_feat_maps.append(fmap1.clone())
            return_feat_maps.append(fmap2.clone())

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(
                fmap1,
                fmap2,
                num_levels=self.args.corr_levels,
                radius=self.args.corr_radius,
                compute_spatial=self.args.compute_spatial,
            )

        if self.return_feat_maps:
            for fm in corr_fn.get_corr_pyramid():
                return_feat_maps.append(fm.permute(1, 0, 2, 3))
        if self.return_feat_maps and self.args.compute_spatial:
            return_feat_maps.append(corr_fn.get_spatial_corr())

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.no_separate_context:
                cnet = self.conv_redir(fmap1)
            else:
                cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        if self.return_feat_maps:
            return_feat_maps.append(net.clone())
            return_feat_maps.append(inp.clone())

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            if self.return_feat_maps:
                return_feat_maps.append(corr.clone())

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                (
                    net,
                    up_mask,
                    delta_flow,
                    motion_features,
                    cor1,
                    cor,
                    cor_flo,
                ) = self.update_block(net, inp, corr, flow)

            if self.return_feat_maps:
                return_feat_maps.append(net.clone())
                return_feat_maps.append(motion_features.clone())
                return_feat_maps.append(cor1.clone())
                return_feat_maps.append(cor.clone())
                return_feat_maps.append(cor_flo.clone())

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            if self.return_feat_maps:
                return_feat_maps.append(flow_up.clone())

            flow_predictions.append(flow_up)

        if test_mode:
            if self.return_feat_maps:
                return coords1 - coords0, flow_up, return_feat_maps
            else:
                return coords1 - coords0, flow_up

        return flow_predictions
