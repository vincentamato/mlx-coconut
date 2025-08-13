import random
import os
import numpy as np
import mlx.core as mx

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return str(self.__dict__)


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    mx.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Set random seed to {seed_value}")
