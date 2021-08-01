import torch

from einops import reduce
# from einops import repeat

batch = 10
channel = 3
depth = 30
width = 512
height = 512

rand_tensor = torch.rand((batch, channel, depth, height, width))

print(rand_tensor.size())

output_tensor = reduce(rand_tensor, 'b c d w h -> b d', 'max')

print(output_tensor.size())
