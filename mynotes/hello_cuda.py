print("hello")

import torch
torch.cuda.is_available()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
