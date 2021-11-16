###########################################################
# Initialize
###########################################################
import torch

# global variable: device for torch
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
