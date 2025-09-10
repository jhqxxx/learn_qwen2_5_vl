import torch
import torch.nn as nn

input = torch.randn((512, 3, 2, 14, 14))
conv_3d = nn.Conv3d(3, 1280, (2, 14, 14), (2, 14, 14), bias=False)
print(conv_3d.weight.shape)
out1 = conv_3d.forward(input)
out1 = out1.squeeze(-1).squeeze(-1).squeeze(-1)
print(out1.shape)
print(out1)


conv_3d_weight = conv_3d.weight
input = input.reshape((512, -1))
conv_3d_weight = conv_3d_weight.reshape((1280, -1))
out2 = input @ conv_3d_weight.T
print(out2.shape)
print(out2)

dif_sum = torch.sum(torch.abs(out1-out2)).item()
dif_max = torch.max(torch.abs(out1-out2)).item()
print(f"max dif: {dif_max}, sum dif: {dif_sum}")