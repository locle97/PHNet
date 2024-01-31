import numpy as np
import cv2
import os
import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import rgb_to_lab, lab_to_rgb


def blend(f, b, a):
    return f * a + b * (1 - a)


class PatchedHarmonizer(nn.Module):
    def __init__(self, grid_count=1, init_weights=[0.9, 0.1]):
        super(PatchedHarmonizer, self).__init__()
        self.eps = 1e-8
        # self.weights = torch.nn.Parameter(torch.ones((grid_count, grid_count)), requires_grad=True)
        # self.grid_weights_ = torch.nn.Parameter(torch.FloatTensor(init_weights), requires_grad=True)
        self.grid_weights = torch.nn.Parameter(
            torch.FloatTensor(init_weights), requires_grad=True
        )
        # self.weights.retain_graph = True
        self.grid_count = grid_count

    def lab_shift(self, x, invert=False):
        x = x.float()
        if invert:
            x[:, 0, :, :] /= 2.55
            x[:, 1, :, :] -= 128
            x[:, 2, :, :] -= 128
        else:
            x[:, 0, :, :] *= 2.55
            x[:, 1, :, :] += 128
            x[:, 2, :, :] += 128

        return x

    def get_mean_std(self, img, mask, dim=[2, 3]) -> tuple[torch.Tensor, torch.Tensor]:
        sum = torch.sum(img * mask, dim=dim)  # (B, C)
        num = torch.sum(mask, dim=dim)  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum(((img - mean) * mask) ** 2, dim=dim) / (num + self.eps)
        var = var[:, :, None, None]

        return mean, torch.sqrt(var + self.eps)

    def compute_patch_statistics(self, lab):
        means, stds = [], []
        bs, dx, dy = (
            lab.shape[0],
            lab.shape[2] // self.grid_count,
            lab.shape[3] // self.grid_count,
        )
        for h in range(self.grid_count):
            cmeans, cstds = [], []
            for w in range(self.grid_count):
                ind = [h * dx, (h + 1) * dx, w * dy, (w + 1) * dy]
                if h == self.grid_count - 1:
                    ind[1] = None
                if w == self.grid_count - 1:
                    ind[-1] = None
                m, v = self.compute_mean_var(
                    lab[:, :, ind[0] : ind[1], ind[2] : ind[3]], dim=[2, 3]
                )
                cmeans.append(m)
                cstds.append(v)
            means.append(cmeans)
            stds.append(cstds)

        return means, stds

    def compute_mean_var(self, x, dim=[1, 2]):
        mean = x.float().mean(dim=dim)[:, :, None, None]
        var = torch.sqrt(x.float().var(dim=dim))[:, :, None, None]

        return mean, var

    def forward(self, fg_rgb, bg_rgb, alpha, masked_stats=False):
        bg_rgb = F.interpolate(bg_rgb, size=(fg_rgb.shape[2:]))  # b x C x H x W

        bg_lab = bg_rgb  # self.lab_shift(rgb_to_lab(bg_rgb/255.))
        fg_lab = fg_rgb  # self.lab_shift(rgb_to_lab(fg_rgb/255.))

        if masked_stats:
            self.bg_global_mean, self.bg_global_var = self.get_mean_std(
                img=bg_lab, mask=(1 - alpha)
            )
            self.fg_global_mean, self.fg_global_var = self.get_mean_std(
                img=fg_lab, mask=torch.ones_like(alpha)
            )
        else:
            self.bg_global_mean, self.bg_global_var = self.compute_mean_var(
                bg_lab, dim=[2, 3]
            )
            self.fg_global_mean, self.fg_global_var = self.compute_mean_var(
                fg_lab, dim=[2, 3]
            )

        self.bg_means, self.bg_vars = self.compute_patch_statistics(bg_lab)
        self.fg_means, self.fg_vars = self.compute_patch_statistics(fg_lab)

        fg_harm = self.harmonize(fg_lab)
        # fg_harm = lab_to_rgb(fg_harm)
        bg = F.interpolate(bg_rgb, size=(fg_rgb.shape[2:])) / 255.0

        composite = blend(fg_harm, bg, alpha)

        return composite, fg_harm

    def harmonize(self, fg):
        harmonized = torch.zeros_like(fg)
        dx = fg.shape[2] // self.grid_count
        dy = fg.shape[3] // self.grid_count
        for h in range(self.grid_count):
            for w in range(self.grid_count):
                ind = [h * dx, (h + 1) * dx, w * dy, (w + 1) * dy]
                if h == self.grid_count - 1:
                    ind[1] = None
                if w == self.grid_count - 1:
                    ind[-1] = None
                harmonized[
                    :, :, ind[0] : ind[1], ind[2] : ind[3]
                ] = self.normalize_channel(
                    fg[:, :, ind[0] : ind[1], ind[2] : ind[3]], h, w
                )

        # harmonized = self.lab_shift(harmonized, invert=True)

        return harmonized

    def normalize_channel(self, value, h, w):
        fg_local_mean, fg_local_var = self.fg_means[h][w], self.fg_vars[h][w]
        bg_local_mean, bg_local_var = self.bg_means[h][w], self.bg_vars[h][w]
        fg_global_mean, fg_global_var = self.fg_global_mean, self.fg_global_var
        bg_global_mean, bg_global_var = self.bg_global_mean, self.bg_global_var

        # global2global normalization
        zeroed_mean = value - fg_global_mean
        # (fg_v * div_global_v +  (1-fg_v) * div_v)
        scaled_var = zeroed_mean * (bg_global_var / (fg_global_var + self.eps))
        normalized_global = scaled_var + bg_global_mean

        # local2local normalization
        zeroed_mean = value - fg_local_mean
        # (fg_v * div_global_v +  (1-fg_v) * div_v)
        scaled_var = zeroed_mean * (bg_local_var / (fg_local_var + self.eps))
        normalized_local = scaled_var + bg_local_mean

        return (
            self.grid_weights[0] * normalized_local
            + self.grid_weights[1] * normalized_global
        )

    def normalize_fg(self, value):
        zeroed_mean = (
            value
            - (self.fg_local_mean * self.grid_weights[None, None, :, :, None, None])
            .sum()
            .squeeze()
        )
        # (fg_v * div_global_v +  (1-fg_v) * div_v)
        scaled_var = zeroed_mean * (
            self.bg_global_var / (self.fg_global_var + self.eps)
        )
        normalized_lg = (
            scaled_var
            + (self.bg_local_mean * self.grid_weights[None, None, :, :, None, None])
            .sum()
            .squeeze()
        )

        return normalized_lg


class PatchNormalizer(nn.Module):
    def __init__(
        self, in_channels=3, eps=1e-7, grid_count=1, weights=[0.5, 0.5], init_value=1e-2
    ):
        super(PatchNormalizer, self).__init__()
        self.grid_count = grid_count
        self.eps = eps

        self.weights = nn.Parameter(torch.FloatTensor(weights), requires_grad=True)
        self.fg_var = nn.Parameter(
            init_value * torch.ones(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.fg_bias = nn.Parameter(
            init_value * torch.zeros(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.patched_fg_var = nn.Parameter(
            init_value * torch.ones(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.patched_fg_bias = nn.Parameter(
            init_value * torch.zeros(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.bg_var = nn.Parameter(
            init_value * torch.ones(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.bg_bias = nn.Parameter(
            init_value * torch.zeros(in_channels)[None, :, None, None],
            requires_grad=True,
        )
        self.grid_weights = torch.nn.Parameter(
            torch.ones((in_channels, grid_count, grid_count))[None, :, :, :]
            / (grid_count * grid_count * in_channels),
            requires_grad=True,
        )

    def local_normalization(self, value):
        zeroed_mean = (
            value
            - (self.fg_local_mean * self.grid_weights[None, None, :, :, None, None])
            .sum()
            .squeeze()
        )
        # (fg_v * div_global_v +  (1-fg_v) * div_v)
        scaled_var = zeroed_mean * (
            self.bg_global_var / (self.fg_global_var + self.eps)
        )
        normalized_lg = (
            scaled_var
            + (self.bg_local_mean * self.grid_weights[None, None, :, :, None, None])
            .sum()
            .squeeze()
        )

        return normalized_lg

    def get_mean_std(self, img, mask, dim=[2, 3]):
        sum = torch.sum(img * mask, dim=dim)  # (B, C)
        num = torch.sum(mask, dim=dim)  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum(((img - mean) * mask) ** 2, dim=dim) / (num + self.eps)
        var = var[:, :, None, None]

        return mean, torch.sqrt(var + self.eps)

    def compute_patch_statistics(self, img, mask):
        means, stds = [], []
        bs, dx, dy = (
            img.shape[0],
            img.shape[2] // self.grid_count,
            img.shape[3] // self.grid_count,
        )
        for h in range(self.grid_count):
            cmeans, cstds = [], []
            for w in range(self.grid_count):
                ind = [h * dx, (h + 1) * dx, w * dy, (w + 1) * dy]
                if h == self.grid_count - 1:
                    ind[1] = None
                if w == self.grid_count - 1:
                    ind[-1] = None
                m, v = self.get_mean_std(
                    img[:, :, ind[0] : ind[1], ind[2] : ind[3]],
                    mask[:, :, ind[0] : ind[1], ind[2] : ind[3]],
                    dim=[2, 3],
                )
                cmeans.append(m.reshape(m.shape[:2]))
                cstds.append(v.reshape(v.shape[:2]))
            means.append(torch.stack(cmeans))
            stds.append(torch.stack(cstds))

        return torch.stack(means), torch.stack(stds)

    def compute_mean_var(self, x, dim=[2, 3]):
        mean = x.float().mean(dim=dim)
        var = torch.sqrt(x.float().var(dim=dim))

        return mean, var

    def forward(self, fg, bg, mask):
        self.local_means, self.local_vars = self.compute_patch_statistics(
            bg, (1 - mask)
        )

        bg_mean, bg_var = self.get_mean_std(bg, 1 - mask)
        zeroed_mean = bg - bg_mean
        unscaled = zeroed_mean / bg_var
        bg_normalized = unscaled * self.bg_var + self.bg_bias

        fg_mean, fg_var = self.get_mean_std(fg, mask)
        zeroed_mean = fg - fg_mean
        unscaled = zeroed_mean / fg_var

        mean_patched_back = (
            self.local_means.permute(2, 3, 0, 1) * self.grid_weights
        ).sum(dim=[2, 3])[:, :, None, None]

        normalized = unscaled * bg_var + bg_mean
        patch_normalized = unscaled * bg_var + mean_patched_back

        fg_normalized = normalized * self.fg_var + self.fg_bias
        fg_patch_normalized = (
            patch_normalized * self.patched_fg_var + self.patched_fg_bias
        )

        fg_result = (
            self.weights[0] * fg_normalized + self.weights[1] * fg_patch_normalized
        )
        composite = blend(fg_result, bg_normalized, mask)

        return composite
