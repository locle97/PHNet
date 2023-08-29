import torch
from model import PHNet

class Inference:
    def __init__(self, **kwargs):
        self.rank=0
        self.__dict__.update(kwargs)
        self.model = PHNet(enc_sizes=self.enc_sizes,
                            skips=self.skips,
                            grid_count=self.grid_counts,
                            init_weights=self.init_weights,
                            init_value=self.init_value)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cuda'), strict=True)
        self.model.eval()

    def harmonize(composite: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([composite, mask], 1)
        with torch.no_grad():
            harmonized = self. model(inputs)['harmonized']
            
        return harmonized