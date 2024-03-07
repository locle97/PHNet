from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
from .normalizer import PatchNormalizer, PatchedHarmonizer


class ConvTransposeUp(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        padding=1,
        stride=2,
        activation=None,
    ):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            activation() if activation is not None else nn.Identity(),
        )


class UpsampleShuffle(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.GELU() if activation else nn.Identity(),
            nn.PixelShuffle(2),
        )

    def reset_parameters(self):
        init_subpixel(self[0].weight)
        nn.init.zeros_(self[0].bias)


class UpsampleResize(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_size=None,
        activation=None,
        scale_factor=2.0,
        mode="bilinear",
    ):
        super().__init__(
            (
                nn.Upsample(scale_factor=scale_factor, mode=mode)
                if out_size is None
                else nn.Upsample(out_size, mode=mode)
            ),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=0
            ),  # noqa
            activation() if activation is not None else nn.Identity(),
        )


def conv_bn(
    in_,
    out_,
    kernel_size=3,
    stride=1,
    padding=1,
    activation=nn.ReLU,
    normalization=nn.InstanceNorm2d,
):
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=padding),
        normalization(out_) if normalization is not None else nn.Identity(),
        activation(),
    )


def init_subpixel(weight):
    co, ci, h, w = weight.shape
    co2 = co // 4
    # initialize sub kernel
    k = torch.empty([co2, ci, h, w])
    nn.init.kaiming_uniform_(k)
    # repeat 4 times
    k = k.repeat_interleave(4, dim=0)
    weight.data.copy_(k)


class DownsampleShuffle(nn.Sequential):
    def __init__(self, in_channels):
        assert in_channels % 4 == 0
        super().__init__(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.PixelUnshuffle(2),
        )

    def reset_parameters(self):
        init_subpixel(self[0].weight)
        nn.init.zeros_(self[0].bias)


def conv_bn_elu(in_, out_, kernel_size=3, stride=1, padding=True):
    # conv layer with ELU activation function
    pad = int(kernel_size / 2)
    if padding is False:
        pad = 0
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=pad),
        nn.ELU(),
    )


class Inference_Data(Dataset):
    def __init__(self, img_path):
        self.input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.input_img = cv2.resize(
            self.input_img, (512, 256), interpolation=cv2.INTER_CUBIC
        )
        self.to_tensor = transforms.ToTensor()
        self.data_len = 1

    def __getitem__(self, index):
        self.tensor_img = self.to_tensor(self.input_img)
        return self.tensor_img

    def __len__(self):
        return self.data_len


class MyAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(
            input=x, kernel_size=(inp_size[2], inp_size[3])
        )  # noqa


class SEBlock(nn.Module):
    def __init__(self, channel, reducation=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reducation, channel),
            nn.Sigmoid(),
        )

    def forward(self, x, aux_inp=None):
        b, c, w, h = x.size()

        def scale(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        y1 = self.avg_pool(x).view(b, c)
        y = self.fc(y1).view(b, c, 1, 1)
        r = x * y
        if aux_inp is not None:
            aux_weitghts = MyAdaptiveMaxPool2d(aux_inp.shape[-1] // 8)(aux_inp)
            aux_weitghts = nn.Sigmoid()(aux_weitghts.mean(1, keepdim=True))
            tmp = x * aux_weitghts
            # tmp_img = (tmp - tmp.min()) / (tmp.max() - tmp.min()) # noqa
            r += tmp

        return r


class ConvTransposeUp(nn.Sequential):  # noqa
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        kernel_size=3,
        stride=2,
        padding=1,
        activation=None,
    ):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            norm(out_channels) if norm is not None else nn.Identity(),
            activation() if activation is not None else nn.Identity(),
        )


class SkipConnect(nn.Module):
    """docstring for RegionalSkipConnect"""

    def __init__(self, channel):
        super(SkipConnect, self).__init__()
        self.rconv = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=False)

    def forward(self, feature):
        return F.relu(self.rconv(feature))


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.Sigmoid(),  # noqa
        )

    def forward(self, x):
        return self.attn(x)


class PatchHarmonizerBlock(nn.Module):
    def __init__(self, in_channels=3, grid_count=5):
        super(PatchHarmonizerBlock, self).__init__()
        self.patch_harmonizer = PatchedHarmonizer(grid_count=grid_count)
        self.head = conv_bn(
            in_channels * 2,
            in_channels,
            kernel_size=3,
            padding=1,
            normalization=None,  # noqa
        )

    def forward(self, fg, bg, mask):
        fg_harm, _ = self.patch_harmonizer(fg, bg, mask)

        return self.head(torch.cat([fg, fg_harm], 1))


class PHNet(nn.Module):
    def __init__(
        self,
        enc_sizes=[3, 16, 32, 64, 128, 256, 512],
        skips=True,
        grid_count=[10, 5, 1],
        init_weights=[0.5, 0.5],
        init_value=0.8,
    ):
        super(PHNet, self).__init__()
        self.skips: bool = skips
        self.feature_extractor = PatchHarmonizerBlock(
            in_channels=enc_sizes[0], grid_count=grid_count[1]
        )
        self.encoder = nn.ModuleList(
            [
                conv_bn(enc_sizes[0], enc_sizes[1], kernel_size=4, stride=2),
                conv_bn(enc_sizes[1], enc_sizes[2], kernel_size=3, stride=1),
                conv_bn(enc_sizes[2], enc_sizes[3], kernel_size=4, stride=2),
                conv_bn(enc_sizes[3], enc_sizes[4], kernel_size=3, stride=1),
                conv_bn(enc_sizes[4], enc_sizes[5], kernel_size=4, stride=2),
                conv_bn(enc_sizes[5], enc_sizes[6], kernel_size=3, stride=1),
            ]
        )

        dec_ins = enc_sizes[::-1]
        dec_sizes = enc_sizes[::-1]
        self.start_level = len(dec_sizes) - len(grid_count)
        self.normalizers = nn.ModuleList(
            [
                PatchNormalizer(
                    in_channels=dec_sizes[self.start_level + i],
                    grid_count=count,
                    weights=init_weights,
                    eps=1e-7,
                    init_value=init_value,
                )
                for i, count in enumerate(grid_count)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                ConvTransposeUp(
                    dec_ins[0],
                    dec_sizes[1],
                    norm=nn.BatchNorm2d,
                    kernel_size=3,
                    stride=1,
                    activation=nn.LeakyReLU,
                ),
                ConvTransposeUp(
                    dec_ins[1],
                    dec_sizes[2],
                    norm=nn.BatchNorm2d,
                    kernel_size=4,
                    stride=2,
                    activation=nn.LeakyReLU,
                ),
                ConvTransposeUp(
                    dec_ins[2],
                    dec_sizes[3],
                    norm=nn.BatchNorm2d,
                    kernel_size=3,
                    stride=1,
                    activation=nn.LeakyReLU,
                ),
                ConvTransposeUp(
                    dec_ins[3],
                    dec_sizes[4],
                    norm=None,
                    kernel_size=4,
                    stride=2,
                    activation=nn.LeakyReLU,
                ),
                ConvTransposeUp(
                    dec_ins[4],
                    dec_sizes[5],
                    norm=None,
                    kernel_size=3,
                    stride=1,
                    activation=nn.LeakyReLU,
                ),
                ConvTransposeUp(
                    dec_ins[5],
                    3,
                    norm=None,
                    kernel_size=4,
                    stride=2,
                    activation=None,  # noqa
                ),
            ]
        )

        self.skip = nn.ModuleList([SkipConnect(x) for x in dec_ins])

        self.SE_block = SEBlock(enc_sizes[6])

    def forward(self, img, mask) -> torch.Tensor:
        x = img

        enc_outs = [x]
        x_harm = self.feature_extractor(x * mask, x * (1 - mask), mask)

        masks = [mask.float()]
        for i, down_layer in enumerate(self.encoder):
            x = down_layer(x)
            scale_factor = 1.0 / (pow(2, 1 - i % 2))
            masks.append(F.interpolate(masks[-1], scale_factor=scale_factor))
            enc_outs.append(x)

        x = self.SE_block(x, aux_inp=x_harm)

        masks = masks[::-1]
        for i, (up_layer, enc_out) in enumerate(
            zip(self.decoder, enc_outs[::-1])
        ):  # noqa
            if i >= self.start_level:
                enc_out = self.normalizers[i - self.start_level](
                    enc_out, enc_out, masks[i]
                )
            x = torch.cat([x, enc_out], 1)
            x = self.skip[i](x)
            x = up_layer(x)

        harmonized: torch.Tensor = F.sigmoid(x)

        return harmonized

    def set_requires_grad(
        self, modules=["encoder", "sh_head", "resquare", "decoder"], value=False  # noqa
    ):
        for module in modules:
            attr = getattr(self, module, None)
            if attr is not None:
                attr.requires_grad_(value)
