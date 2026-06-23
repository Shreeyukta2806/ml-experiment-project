"""
Loads images from dataset/<class_name>/*.jpg style folders,
splits into train/val, and returns PyTorch DataLoaders.

Expected folder structure:
    dataset/
        matchbox/   *.jpg
        spoon/      *.jpg
        guitar/     *.jpg
        pen/        *.jpg
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 128  # resize all images to 128x128 (keeps training fast)

# Basic normalization stats (ImageNet means/stds — standard practice even
# for small custom datasets, since they're a reasonable general-purpose default)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(augment=False):
    """
    augment=False -> just resize + normalize (used for validation, and for
                      baseline training runs)
    augment=True  -> adds random flip/rotation (used in the data-augmentation
                      experiment, to see its effect on overfitting)
    """
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))] + aug + base[1:]
        )

    return transforms.Compose(base)


class CustomImageDataset(Dataset):
    def __init__(self, filepaths, labels, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def load_dataset_paths(dataset_dir="dataset"):
    """Walks the dataset folder, returns filepaths + integer labels + class names."""
    class_names = sorted(
        [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    )
    if len(class_names) == 0:
        raise ValueError(f"No class folders found in {dataset_dir}/")

    filepaths = []
    labels = []
    valid_ext = (".jpg", ".jpeg", ".png")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_ext):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    return filepaths, labels, class_names


def get_dataloaders(dataset_dir="dataset", batch_size=16, augment=False, val_split=0.2, seed=42):
    """
    Returns (train_loader, val_loader, class_names).

    val_split=0.2 means 20% of images held out for validation —
    these images are never trained on, only used to check generalization.
    """
    filepaths, labels, class_names = load_dataset_paths(dataset_dir)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,  # ensures each class is proportionally represented in both splits
    )

    train_dataset = CustomImageDataset(train_paths, train_labels, get_transforms(augment=augment))
    val_dataset = CustomImageDataset(val_paths, val_labels, get_transforms(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes found: {class_names}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    # quick sanity check — run this once your dataset/ folder has images in it
    train_loader, val_loader, class_names = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Batch image tensor shape:", images.shape)
    print("Batch labels:", labels)
