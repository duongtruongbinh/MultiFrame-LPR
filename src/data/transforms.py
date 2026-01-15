"""Augmentation pipelines for training, validation, and degradation."""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Training augmentation pipeline with geometric and color transforms."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-5, 5),
            fill=128,
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.Rotate(limit=10, p=0.3), 
        A.ChannelShuffle(p=0.3),
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            p=0.3
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def get_light_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Light training pipeline: resize + normalize only."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_degradation_transforms() -> A.Compose:
    """Pipeline to convert HR images to synthetic LR."""
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0)
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
        ], p=0.7),
        A.ImageCompression(quality_range=(20, 50), p=0.5),
        A.Downscale(scale_range=(0.3, 0.5), p=0.5),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Validation transform pipeline (resize + normalize only)."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
