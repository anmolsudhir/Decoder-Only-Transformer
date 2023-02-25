import torch
import torch.nn as nn
from torch.nn import functional as F

x = torch.randn(3, 2, 2)
print(x)
x = x/2
print(x)