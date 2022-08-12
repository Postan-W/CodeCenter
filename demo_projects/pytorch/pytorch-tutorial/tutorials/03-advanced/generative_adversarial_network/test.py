import torch
import os
from torchvision.utils import save_image
data = torch.randn(100, 64).to('cuda')
model = torch.load("./G.pth")
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
fake_images = model(data)
fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
save_image(denorm(fake_images),"result1.png")

