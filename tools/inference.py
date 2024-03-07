import torch
from .model import PHNet
import torchvision.transforms.functional as tf
from .util import log


class Inference:
    def __init__(self, **kwargs):
        self.rank = 0
        self.__dict__.update(kwargs)
        self.model = PHNet(
            enc_sizes=self.enc_sizes,
            skips=self.skips,
            grid_count=self.grid_counts,
            init_weights=self.init_weights,
            init_value=self.init_value,
        )
        state = torch.load(self.checkpoint.harmonizer, map_location=self.device)  # noqa

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def harmonize(self, composite, mask):
        if len(composite.shape) < 4:
            composite = composite.unsqueeze(0)
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)
        composite: torch.Tensor = tf.resize(
            composite, [self.input.image_size, self.input.image_size]
        )
        mask: torch.Tensor = tf.resize(mask, [self.input.image_size, self.input.image_size])  # noqa

        log(composite.shape, mask.shape)
        with torch.no_grad():
            harmonized = self.model(composite, mask)  # ['harmonized']

        result = harmonized * mask + composite * (1 - mask)

        return result
