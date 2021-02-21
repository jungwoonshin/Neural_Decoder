import torch

# concat2 =  concat1[:,torch.randperm(concat1.shape[1])]


#         concat3 =  concat1[:,torch.randperm(concat1.shape[1])]

a= torch.arange(0,12).reshape(4,3)
b = a[:,torch.randperm(a.shape[1])]
print(a)
print(b)