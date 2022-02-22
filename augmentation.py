import torch
from torchvision import transforms


basic_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
])