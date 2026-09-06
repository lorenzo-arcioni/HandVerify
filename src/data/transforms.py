from torchvision import transforms


def get_train_transforms(target_size: int = 448, aug: bool = False):
    """
    Get training transforms, con augmentation opzionale.

    Args:
        target_size: Target image size
        aug: se True, aggiunge augmentation geometriche e fotometriche
             (oltre a RandomResizedCrop + RandomRotation gia' presenti).
             Se False (default), comportamento invariato rispetto a prima.
    Returns:
        Composed transforms
    """
    base = [
        transforms.RandomResizedCrop(target_size, scale=(0.9, 1.1)),
        transforms.RandomRotation(15),
    ]

    if aug:
        base += [
            # --- Geometriche aggiuntive (leggere) ---
            transforms.RandomAffine(
                degrees=0,              # rotazione gia' gestita sopra
                translate=(0.03, 0.03), # shift minimo
                shear=5,                # taglio leggero
            ),
            # --- Fotometriche (leggere) ---
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.3,
            ),
        ]

    base.append(transforms.ToTensor())
    return transforms.Compose(base)


def get_test_transforms(target_size: int = 448):
    """
    Get test/validation transforms (no augmentation).
    Args:
        target_size: Target image size
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])