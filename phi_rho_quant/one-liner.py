from torchao.quantization import quantize_
from torchao.quantization.prototype.phi_rho import PhiRhoCodebookConfig
import torch, torch.nn as nn

m = nn.Linear(4096, 4096, bias=False)
quantize_(m, PhiRhoCodebookConfig(num_codebook_entries=256))