import torch
import numpy as np
import random
from pytorch_msssim import SSIM as ssim  # у меня значение SSIM не совпало с kornia


class MSE(object):
    """MSE."""

    def __init__(self):
        """Инициализация объекта. Кумулятивных полей нет."""
        pass

    def __call__(
        self, pred: torch.tensor, target: torch.tensor, mask: torch.tensor
    ) -> float:
        """Подсчет MSE.

        Args:
            pred (torch.tensor): Прогноз модели. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                 Если тип не соответствует float32, то pred к нему приводится.
            target (torch.tensor): GT. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                   Если тип не соответствует float32, то target к нему приводится.
            mask (torch.tensor): Маска fg области. В MSE не используется.

        Returns:
            mse (float): Значение метрики MSE.
        """
        if pred.dtype != torch.float32:
            pred = pred.to(torch.float32)

        if target.dtype != torch.float32:
            target = target.to(torch.float32)

        return ((pred - target) ** 2).mean().item()


class fMSE(object):
    """
    Метрика, используемая для подсчета MSE в fg области.
    Reference: https://github.com/SamsungLabs/image_harmonization/blob/4d16b1257eaae115c8714bf147f3113a9dabdf51/iharm/inference/metrics.py#L83C7-L83C7.
    """

    def __init__(self, eps: float = 1e-6):
        """Инициализация объекта. Кумулятивных полей нет.

        Args:
            eps (float): Для числовой стабильности в знаменатель. По умолчанию 1e-6.
        """
        self.eps = eps

    def __call__(
        self, pred: torch.tensor, target: torch.tensor, mask: torch.tensor
    ) -> float:
        """Подсчет fMSE.

        Args:
            pred (torch.tensor): Прогноз модели. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                 Если тип не совпадает - pred приводится во float32.
            target (torch.tensor): GT. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                   Если тип не совпадает - target приводится во float32.
            mask (torch.tensor): Маска fg области. Должна быть в формате int32 и состоять только из нулей и единиц.
                                 Содержание проверяется, иначе - exception.
                                 Если тип не совпадает - маска приводится к int32 принудительно.
                                 Shape - (h, w). Тип - int32. Unique values - [0, 1].

        Returns:
            fMSE (float): Значение метрики fMSE.
        """

        combined = torch.cat((mask.unique(), torch.tensor([0, 1]).to(mask.device)))
        uniques, counts = combined.unique(return_counts=True)
        # difference = uniques[counts == 1]
        if len(uniques) > 2:
            raise ValueError(
                "Test/val mask is non-binary. Give mask from dataset via nearest interpolation applied on rounded mask."
            )

        if mask.dtype != torch.int32:
            mask = mask.to(torch.int32)

        if pred.dtype != torch.float32:
            pred = pred.to(torch.float32)

        if target.dtype != torch.float32:
            target = target.to(torch.float32)

        diff = mask.unsqueeze(2) * ((pred - target) ** 2)
        return (
            diff.sum() / (diff.size(2) * mask.sum() + self.eps)
        ).item()  # diff.size(2) == [channels] == 3.


class PSNR(MSE):
    """
    PSNR.
    Reference: https://github.com/SamsungLabs/image_harmonization/blob/4d16b1257eaae115c8714bf147f3113a9dabdf51/iharm/inference/metrics.py#L89C17-L89C17.
    """

    def __init__(self, eps: float = 1e-6):
        """Инициализация объекта. Кумулятивных полей нет.

        Args:
            eps (float): Для числовой стабильности в знаменатель. По умолчанию 1e-6.
        """
        super().__init__()
        self.eps = eps

    def __call__(
        self, pred: torch.tensor, target: torch.tensor, mask: torch.tensor
    ) -> float:
        """Подсчет PSNR.

        Args:
            pred (torch.tensor): Прогноз модели. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                 Если тип не совпадает - pred приводится во float32 (в MSE).
            target (torch.tensor): GT. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                   Если тип не совпадает - target приводится во float32 (в MSE).
            mask (torch.tensor): Маска fg области. В PSNR не используется.

        Returns:
            PSNR (float): Значение метрики PSNR.
        """
        msev = super().__call__(pred=pred, target=target, mask=mask)
        squared_maxv = target.to(torch.float32).max().item() ** 2

        return 10 * np.log10(squared_maxv / (msev + self.eps))


class fPSNR(fMSE):
    """
    fPSNR. Такой же как PSNR, но вычисляет не MSE, а fMSE.
    """

    def __init__(self, eps: float = 1e-6):
        """Инициализация объекта. fPSNR вычисляет fMSE.

        Args:
            eps (float): Для числовой стабильности в знаменатель. По умолчанию 1e-6.
        """
        super().__init__(eps=eps)
        self.eps = eps

    def __call__(
        self, pred: torch.tensor, target: torch.tensor, mask: torch.tensor
    ) -> float:
        """Подсчет fPSNR.

        Args:
            pred (torch.tensor): Прогноз модели. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                 Если тип не совпадает - pred приводится во float32 (в fMSE).
            target (torch.tensor): GT. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                   Если тип не совпадает - target приводится во float32 (в fMSE).
            mask (torch.tensor): Маска fg области. Должна быть в формате int32 и состоять только из нулей и единиц.
                                 Внутри fMSE cодержание проверяется, иначе - exception.
                                 Если тип не совпадает - маска приводится к int32 принудительно.
                                 Shape - (h, w). Тип - int32. Unique values - [0, 1].

        Returns:
            fPSNR (float): Значение fPSNR метрики.
        """
        fmsev = super().__call__(pred=pred, target=target, mask=mask)
        squared_maxv = target.to(torch.float32).max().item() ** 2

        return 10 * np.log10(squared_maxv / (fmsev + self.eps))


class SSIM(object):
    """Метрика SSIM.
    Используется framework pytorch_msssim.
    """

    def __init__(self):
        """Инициализация объекта. Кумулятивных полей нет."""
        self.ssim = ssim(data_range=255.0, channel=3, size_average=True)

    def __call__(
        self, pred: torch.tensor, target: torch.tensor, mask: torch.tensor
    ) -> float:
        """Подсчет SSIM.

        Args:
            pred (torch.tensor): Прогноз модели. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                 Если тип не совпадает - pred приводится во float32.
            target (torch.tensor): GT. Shape - (h, w, 3). Тип - float32. Range - (0; 255).
                                   Если тип не совпадает - pred приводится во float32.
            mask (torch.tensor): Маска fg области. В SSIM не используется.

        Returns:
            ssim (float): Значение метрики SSIM.
        """
        if pred.dtype != torch.float32:
            pred = pred.to(torch.float32)

        if target.dtype != torch.float32:
            target = target.to(torch.float32)

        # (h, w, 3) -> (3, h, w) -> (1, 3, h, w)
        return self.ssim(
            pred.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0)
        ).item()


if __name__ == "__main__":
    # seeds
    seed = 1001
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    pred = torch.rand(256, 256, 3, dtype=torch.float32) * 255
    gt = torch.rand(256, 256, 3, dtype=torch.float32) * 255
    mask = torch.randint(
        low=0, high=1 + 1, size=(512, 512), dtype=torch.uint8
    )  # {0, 1}
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest"
    )[
        0, 0
    ]  # беру h, w

    for metric in [MSE(), fMSE(), PSNR(), fPSNR(), SSIM()]:
        metricv = metric(
            pred=pred.to("cuda:0"), target=gt.to("cuda:0"), mask=mask.to("cuda:0")
        )
        print(f"{metric.__class__.__name__}: {metricv}")
