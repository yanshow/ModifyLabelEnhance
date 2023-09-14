import torch
a=torch.tensor([1,2],dtype=torch.int32)
print(a)
a=a.to(torch.long)
print(a)