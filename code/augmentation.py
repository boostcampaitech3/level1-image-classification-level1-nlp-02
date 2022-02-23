import torchvision
from torchvision import transforms, utils

basic_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
