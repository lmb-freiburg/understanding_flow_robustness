import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from .submodules import conv, correlate, deconv, predict_flow

"Parameter count , 39,175,298 "


class FlowNetC(nn.Module):
    def __init__(self, batchNorm=False, div_flow=20, return_feat_maps=False):
        super().__init__()
        # self.training = training

        self.rgb_max = 1
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.return_feat_maps = return_feat_maps

        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)

        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)
        self.corr = correlate  # Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)

        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

        self.upsample1 = nn.Upsample(scale_factor=4, mode="bilinear")

    @staticmethod
    def normalize(im):
        im = im - 0.5
        im = im / 0.5
        return im

    def normalize_correctly(self, im):  # pylint: disable=no-self-use
        mean = np.array([0.40066648, 0.39482617, 0.3784785])  # RGB
        # mean=np.array([0.3784785,0.39482617,0.40066648]) # BGR
        std = np.array([1, 1, 1])
        return (
            im - torch.from_numpy(mean[None, :, None, None]).cuda()
        ) / torch.from_numpy(std[None, :, None, None]).cuda()

    def forward(self, x1, x2, overwrite_feat_maps=None):
        # inputs = torch.cat((x1, x2), dim=1)
        # rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,))
        # #print(rgb_mean.size())

        # x = (inputs - rgb_mean) / self.rgb_max
        # x1 = x[:,:2,:,:]
        # x2 = x[:,3:,:,:]

        # x1 = self.normalize(x1)
        # x2 = self.normalize(x2)

        x1 = self.normalize_correctly(x1).float()
        x2 = self.normalize_correctly(x2).float()

        if self.return_feat_maps:
            return_feat_maps = []

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv1a.clone())
        out_conv2a = self.conv2(out_conv1a)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv2a.clone())
        out_conv3a = self.conv3(out_conv2a)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv3a.clone())

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv1b.clone())
        out_conv2b = self.conv2(out_conv1b)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv2b.clone())
        out_conv3b = self.conv3(out_conv2b)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv3b.clone())

        if (
            overwrite_feat_maps is not None
            and "conv3a" in overwrite_feat_maps.keys()
            and "conv3b" in overwrite_feat_maps.keys()
        ):
            out_conv3a = (
                torch.from_numpy(overwrite_feat_maps["conv3a"]).unsqueeze(0).cuda()
            )
            out_conv3b = (
                torch.from_numpy(overwrite_feat_maps["conv3b"]).unsqueeze(0).cuda()
            )

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        if self.return_feat_maps:
            return_feat_maps.append(out_corr.clone())
        if not overwrite_feat_maps is None and "corr" in overwrite_feat_maps.keys():
            out_corr = torch.from_numpy(overwrite_feat_maps["corr"]).unsqueeze(0).cuda()
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)
        if not overwrite_feat_maps is None and "conv_redir" in overwrite_feat_maps.keys():
            out_conv_redir = (
                torch.from_numpy(overwrite_feat_maps["conv_redir"]).unsqueeze(0).cuda()
            )

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)
        if not overwrite_feat_maps is None and "conv3_1" in overwrite_feat_maps.keys():
            out_conv3_1 = (
                torch.from_numpy(overwrite_feat_maps["conv3_1"]).unsqueeze(0).cuda()
            )

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return (
                self.upsample1(flow2 * self.div_flow),
                self.upsample1(flow3 * self.div_flow),
                self.upsample1(flow4 * self.div_flow),
                self.upsample1(flow5 * self.div_flow),
                self.upsample1(flow6 * self.div_flow),
            )
        else:
            if self.return_feat_maps:
                return self.upsample1(flow2 * self.div_flow), return_feat_maps
            else:
                return self.upsample1(flow2 * self.div_flow)
