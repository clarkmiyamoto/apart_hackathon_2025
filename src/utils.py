import torch
import numpy as np
import random

def set_seed(seed: int = 42) -> torch.Generator:
    """Make Python, NumPy, PyTorch (CPU & CUDA) reproducible and
    return a torch.Generator seeded identically (handy for DataLoader)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # if you use CUDA
    torch.backends.cudnn.deterministic = True  # slower but deterministic
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)