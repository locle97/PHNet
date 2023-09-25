import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from itertools import chain

from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation
device = 'cpu'


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        prefix=nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prefix = prefix

    def forward(self, x):
        if self.prefix is not None:
            x = self.prefix(x)
        x = self.encoder(x)["hidden_states"]  # transformers
        return self.decoder(x)


def conv2d_relu(input_filters, output_filters, kernel_size=3,  bias=True):
    return nn.Sequential(
        nn.Conv2d(input_filters, output_filters,
                  kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(output_filters)
    )


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2*fpn_out, fpn_out, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):

        features[:-1] = [conv1x1(feature) for feature,
                         conv1x1 in zip(features[:-1], self.conv1x1)]
        feature = up_and_add(self.smooth_conv[0](features[0]), features[1])
        feature = up_and_add(self.smooth_conv[1](feature), features[2])
        feature = up_and_add(self.smooth_conv[2](feature), features[3])

        H, W = features[-1].size(2), features[-1].size(3)
        x = [feature, features[-1]]
        x = [F.interpolate(x_el, size=(H, W), mode='bilinear',
                           align_corners=True) for x_el in x]

        x = self.conv_fusion(torch.cat(x, dim=1))
        # x = F.interpolate(x, size=(H*4, W*4), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class UperNet_swin(nn.Module):
    # Implementing only the object path
    def __init__(self, backbone, pretrained=True):
        super(UperNet_swin, self).__init__()

        self.backbone = backbone
        feature_channels = [192, 384, 768, 768]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        features = self.backbone(x)["hidden_states"]
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels=(3, 192, 384, 768, 768),
        decoder_channels=(512, 256, 128, 64),
        n_blocks=4,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        upscale_factor = 4
        self.matting_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=upscale_factor),
        )

    def preprocess_features(self, x):
        features = []
        for out_tensor in x:
            bs, n, f = out_tensor.size()
            h = int(n**0.5)
            feature = out_tensor.view(-1, h, h,
                                      f).permute(0, 3, 1, 2).contiguous()
            features.append(feature)
        return features

    def forward(self, features):
        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        features = self.preprocess_features(features)

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            # y_i = self.upsample1(y_i)
        # hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        x = self.matting_head(x)
        x = 1-nn.ReLU()(1-x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = conv2d_relu(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3
        )
        self.conv2 = conv2d_relu(
            out_channels,
            out_channels,
            kernel_size=3,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

    def forward(self, x, skip=None):
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # print(x.shape,skip.shape)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv1 = conv2d_relu(
            in_channels,
            out_channels,
            kernel_size=3,
        )
        conv2 = conv2d_relu(
            out_channels,
            out_channels,
            kernel_size=3,
        )
        super().__init__(conv1, conv2)


class SegForm(nn.Module):
    def __init__(self):
        super(SegForm, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True
                                                                      )

    def forward(self, image):
        img_segs = self.model(image)
        upsampled_logits = nn.functional.interpolate(img_segs.logits,
                                                     scale_factor=4,
                                                     mode='nearest',
                                                     )
        return upsampled_logits


class StyleMatte(nn.Module):
    def __init__(self):
        super(StyleMatte, self).__init__()
        self.fpn = FPN_fuse(feature_channels=[256, 256, 256, 256], fpn_out=256)
        self.pixel_decoder = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance").base_model.pixel_level_module
        self.fgf = FastGuidedFilter()
        self.conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, image, normalize=False):
        decoder_out = self.pixel_decoder(image)
        decoder_states = list(decoder_out.decoder_hidden_states)
        decoder_states.append(decoder_out.decoder_last_hidden_state)
        out_pure = self.fpn(decoder_states)

        image_lr = nn.functional.interpolate(image.mean(1, keepdim=True),
                                             scale_factor=0.25,
                                             mode='bicubic',
                                             align_corners=True
                                             )
        out = self.conv(out_pure)
        out = self.fgf(image_lr, out, image.mean(
            1, keepdim=True))

        return torch.sigmoid(out)

    def get_training_params(self):
        return list(self.fpn.parameters())+list(self.conv.parameters())


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter((x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


class FastGuidedFilter(nn.Module):
    def __init__(self, r=1, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        # N
        N = self.boxfilter(lr_x.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))

        # mean_x
        mean_x = self.boxfilter(lr_x) / N
        # mean_y
        mean_y = self.boxfilter(lr_y) / N
        # cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = F.interpolate(
            A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(
            b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b


class DeepGuidedFilterRefiner(nn.Module):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2d(
            4, 4, kernel_size=3, padding=1, bias=False, groups=4)
        self.box_filter.weight.data[...] = 1 / 9
        self.conv = nn.Sequential(
            nn.Conv2d(4 * 2 + hid_channels, hid_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, 4, kernel_size=1, bias=True)
        )

    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)

        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x = self.box_filter(base_x * base_x) - mean_x * mean_x

        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        b = mean_y - A * mean_x

        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)

        out = A * fine_x + b
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :,        -1:] - input[:, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :,        -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
