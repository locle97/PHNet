import torch
import numpy as np
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn

import kornia.metrics as metrics


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) +
                      torch.pow(Dgb, 2), 0.5)

        return k


def get_image_gradients(image):
    """Returns image gradients (dy, dx) for each color channel.
    Both output tensors have the same shape as the input: [b, c, h, w]. 
    Places the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y). 
    That means that dy will always have zeros in the last row,
    and dx will always have zeros in the last column.
    This can be used to implement the anisotropic 2-D version of the 
    Total Variation formula:
        https://en.wikipedia.org/wiki/Total_variation_denoising
    (anisotropic is using l1, isotropic is using l2 norm)

    Arguments:
        image: Tensor with shape [b, c, h, w].
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal image
        gradients (1-step finite difference).  
    Raises:
      ValueError: If `image` is not a 3D image or 4D tensor.
    """

    image_shape = image.shape

    if len(image_shape) == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        dx = image[:, 1:, :] - image[:, :-1, :]  # pixel_dif2, f_v_1-f_v_2
        dy = image[1:, :, :] - image[:-1, :, :]  # pixel_dif1, f_h_1-f_h_2

    elif len(image_shape) == 4:
        # Return tensors with same size as original image
        # adds one pixel pad to the right and removes one pixel from the left
        right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
        # adds one pixel pad to the bottom and removes one pixel from the top
        bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :]

        # right and bottom have the same dimensions as image
        dx, dy = right - image, bottom - image

        # this is required because otherwise results in the last column and row having
        # the original pixels from the image
        # dx will always have zeros in the last column, right-left
        dx[:, :, :, -1] = 0
        # dy will always have zeros in the last row,    bottom-top
        dy[:, :, -1, :] = 0
    else:
        raise ValueError(
            'image_gradients expects a 3D [h, w, c] or 4D tensor '
            '[batch_size, c, h, w], not %s.', image_shape)

    return dy, dx


def get_4dim_image_gradients(image):
    # Return tensors with same size as original image
    # Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    # adds one pixel pad to the right and removes one pixel from the left
    right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
    # adds one pixel pad to the bottom and removes one pixel from the top
    bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :]
    # displaces in diagonal direction
    botright = F.pad(image, [0, 1, 0, 1])[..., 1:, 1:]

    # right and bottom have the same dimensions as image
    dx, dy = right - image, bottom - image
    dn, dp = botright - image, right - bottom
    # dp is positive diagonal (bottom left to top right)
    # dn is negative diagonal (top left to bottom right)

    # this is required because otherwise results in the last column and row having
    # the original pixels from the image
    dx[:, :, :, -1] = 0  # dx will always have zeros in the last column, right-left
    dy[:, :, -1, :] = 0  # dy will always have zeros in the last row,    bottom-top
    dp[:, :, -1, :] = 0  # dp will always have zeros in the last row

    return dy, dx, dp, dn


class GradientLoss(nn.Module):
    def __init__(self, loss_f=None, reduction='mean', gradientdir='2d'):  # 2d or 4d
        super(GradientLoss, self).__init__()
        self.criterion = loss_f
        self.gradientdir = gradientdir

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.gradientdir == '4d':
            inputdy, inputdx, inputdp, inputdn = get_4dim_image_gradients(x)
            targetdy, targetdx, targetdp, targetdn = get_4dim_image_gradients(
                y)
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy) +
                    self.criterion(inputdp, targetdp) + self.criterion(inputdn, targetdn))/4
            # input_grad = torch.pow(torch.pow((inputdy) * 0.25, 2) + torch.pow((inputdx) * 0.25, 2) \
            #            + torch.pow((inputdp) * 0.25, 2) + torch.pow((inputdn) * 0.25, 2), 0.5)
            # target_grad = torch.pow(torch.pow((targetdy) * 0.5, 2) + torch.pow((targetdx) * 0.5, 2) \
            #            + torch.pow((targetdp) * 0.25, 2) + torch.pow((targetdn) * 0.25, 2), 0.5)
            # return self.criterion(input_grad, target_grad)
        else:  # '2d'
            inputdy, inputdx = get_image_gradients(x)
            targetdy, targetdx = get_image_gradients(y)
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy))/2
            # input_grad = torch.pow(torch.pow((inputdy) * 0.5, 2) + torch.pow((inputdx) * 0.5, 2), 0.5)
            # target_grad = torch.pow(torch.pow((targetdy) * 0.5, 2) + torch.pow((targetdx) * 0.5, 2), 0.5)
            # return self.criterion(input_grad, target_grad)


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes the PSNR loss.
    Args:
        input: the input image with shape :math:`(*)`.
        labels : the labels image with shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.
    """

    return -1.0 * metrics.psnr(input, target, max_val)


class PSNRLoss(nn.Module):
    r"""Create a criterion that calculates the PSNR loss.
    Args:
        max_val: The maximum value in the input tensor.

    Shape:
        - Input: arbitrary dimensional tensor :math:`(*)`.
        - Target: arbitrary dimensional tensor :math:`(*)` same shape as input.
        - Output: a scalar.
    """

    def __init__(self, max_val: float) -> None:
        super().__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr_loss(input, target, self.max_val)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(
                224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(
                224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def gauss_kernel(size=5, device=torch.device('cpu'), channels=1):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1],
                   x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1],
                   x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr


class LaplacianLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

    def forward(self, input, target):
        pyr_input = laplacian_pyramid(
            img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(
            img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

# def matting_loss(pred_fgr, pred_pha, true_fgr, true_pha):
#     """
#     Args:
#         pred_fgr: Shape(B, T, 3, H, W)
#         pred_pha: Shape(B, T, 1, H, W)
#         true_fgr: Shape(B, T, 3, H, W)
#         true_pha: Shape(B, T, 1, H, W)
#     """
#     loss = dict()
#     # Alpha losses
#     loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
#     loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
#     loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
#                                        true_pha[:, 1:] - true_pha[:, :-1]) * 5
#     # Foreground losses
#     true_msk = true_pha.gt(0)
#     pred_fgr = pred_fgr * true_msk
#     true_fgr = true_fgr * true_msk
#     loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
#     loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
#                                        true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
#     # Total
#     loss['total'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian'] \
#                   + loss['fgr_l1'] + loss['fgr_coherence']
#     return loss


def segmentation_loss(pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)


def calculate_sad_mse_mad(predict_old, alpha, trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel == 0:
        pixel = trimap.shape[0]*trimap.shape[1] - \
            float((trimap == 255).sum())-float((trimap == 0).sum())
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff


def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0]*predict.shape[1]
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff


def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict-alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    weight_fg[trimap == 255] = 1.
    weight_bg[trimap == 0] = 1.
    weight_trimap[trimap == 128] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000
    return sad_fg, sad_bg


def compute_gradient_whole_image(pd, gt):
    from scipy.ndimage import gaussian_filter
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map) / 10
    return loss


def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
    from scipy.ndimage import morphology
    from skimage.measure import label, regionprops
    h, w = pd.shape
    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords
        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]
        dist_maps = morphology.distance_transform_edt(omega == 0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
    return loss
