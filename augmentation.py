import torch
from torchvision import transforms


basic_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
])