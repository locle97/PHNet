import torch
from model import PHNet
import torchvision.transforms.functional as tf

class Inference:
    def __init__(self, **kwargs):
        self.rank=0
        self.__dict__.update(kwargs)
        print(self.__dict__)
        self.model = PHNet(enc_sizes=self.enc_sizes,
                            skips=self.skips,
                            grid_count=self.grid_counts,
                            init_weights=self.init_weights,
                            init_value=self.init_value)
        state = torch.load(self.checkpoint, map_location=self.device)
  
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        
    def harmonize(self, composite, mask):
        if len(composite.shape) < 4:
            composite = composite.unsqueeze(0)
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)
        composite = tf.resize(composite, [self.input.image_size, self.input.image_size])
        mask = tf.resize(mask, [self.input.image_size, self.input.image_size])

        with torch.no_grad():
            harmonized = self.model(composite, mask)['harmonized']

        return harmonized