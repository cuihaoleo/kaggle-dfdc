import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.ChannelShuffle(p=0.1),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(p=0.1),
    A.HueSaturationValue(p=0.1),
    A.IAAAdditiveGaussianNoise(p=0.1),
    A.IAASharpen(p=0.5),
    A.ISONoise(p=0.3),
    A.RandomBrightness(p=0.8),
    A.RandomBrightnessContrast(p=0.2),
    A.ToSepia(p=0.1),
    A.Resize(299, 299),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(299, 299),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2(),
])
