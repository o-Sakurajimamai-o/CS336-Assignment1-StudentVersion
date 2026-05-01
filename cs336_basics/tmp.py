import torch
print(torch.cuda.is_available())       # 应该是 True
print(torch.cuda.get_device_name(0))   # 应该显示 RTX 5070