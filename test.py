from trainer import Trainer, Inference
from omegaconf import OmegaConf
from torch import multiprocessing as mp
from torch import distributed as dist
import os
import sys
        
if __name__ == '__main__':

    args = OmegaConf.load(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), f"config/test.yaml"))
    
    infer = Inference(**args)
    