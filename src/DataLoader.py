import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2


class BMDDataset(Dataset):
    """BMD prediction dataset"""

    def __init__(self, file_paths, labels, img_size=256, label_mean=None, label_std=None):
        """
        Args:
            file_paths: List of NIfTI file paths
            labels: List of BMD labels
            img_size: Output image size
            label_mean: Mean for label standardization
            label_std: Std for label standardization
        """
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load NIfTI file
        nii_img = nib.load(self.file_paths[idx])
        data = nii_img.get_fdata()

        # Get middle 15 slices
        depth = data.shape[2]
        center = depth // 2
        num_slices = 15
        half_slices = num_slices // 2  # 7

        start_idx = center - half_slices
        end_idx = start_idx + num_slices

        # Handle edge cases (if depth < 15)
        if depth < num_slices:
            # Pad with zeros if not enough slices
            slices = np.zeros((self.img_size, self.img_size, num_slices))
            padding = (num_slices - depth) // 2
            for i in range(depth):
                slice_2d = cv2.resize(data[:, :, i], (self.img_size, self.img_size))
                slices[:, :, padding + i] = slice_2d
        else:
            # Extract middle 15 slices
            slices = np.zeros((self.img_size, self.img_size, num_slices))
            for i in range(num_slices):
                slice_2d = cv2.resize(data[:, :, start_idx + i], (self.img_size, self.img_size))
                slices[:, :, i] = slice_2d

        # Normalize (Z-score standardization)
        mean_val = slices.mean()
        std_val = slices.std()
        if std_val > 0:
            slices = (slices - mean_val) / std_val
        else:
            slices = np.zeros_like(slices)

        # Convert to PyTorch tensor [15, H, W] - 15 channels
        image_tensor = torch.FloatTensor(slices).permute(2, 0, 1)  # [H, W, 15] -> [15, H, W]

        # Standardize label if mean/std provided
        label = self.labels[idx]
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std
        label_tensor = torch.FloatTensor([label])

        return image_tensor, label_tensor.squeeze()


def load_metadata(xlsx_path):
    """Load metadata.xlsx, return ID to BMD mapping"""
    df = pd.read_excel(xlsx_path)
    return dict(zip(df['ID'].astype(int), df['BMD'].astype(float)))


def get_patient_files(data_dir):
    """
    Scan directory, build patient ID to file path mapping
    Filename format: {PatientID}_{Date}_{Description}.nii.gz
    """
    patient_files = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.nii.gz'):
            # Parse patient ID (number before first _ or @)
            try:
                # Handle filenames like "368@_..." or "368_..."
                first_part = filename.split('_')[0]
                # Remove @ if present
                first_part = first_part.replace('@', '')
                patient_id = int(first_part)
                file_path = os.path.join(data_dir, filename)

                if patient_id not in patient_files:
                    patient_files[patient_id] = []
                patient_files[patient_id].append(file_path)
            except ValueError:
                print(f"Warning: Cannot parse filename {filename}")
                continue

    return patient_files


def create_dataloaders(data_dir, xlsx_path, batch_size=16, img_size=256, random_state=42):
    """
    Create train/val/test DataLoaders

    Args:
        data_dir: NIfTI files directory
        xlsx_path: metadata.xlsx path
        batch_size: Batch size
        img_size: Image size
        random_state: Random seed

    Returns:
        train_loader, val_loader, test_loader, label_stats (dict with mean and std)
    """
    # Load metadata
    bmd_dict = load_metadata(xlsx_path)

    # Get file mapping
    patient_files = get_patient_files(data_dir)

    # Build (file_path, bmd) pairs
    file_paths = []
    labels = []

    for patient_id, files in patient_files.items():
        if patient_id in bmd_dict:
            bmd = bmd_dict[patient_id]
            for file_path in files:
                file_paths.append(file_path)
                labels.append(bmd)
        else:
            print(f"Warning: Patient ID {patient_id} not found in metadata")

    print(f"Total samples loaded: {len(file_paths)}")

    # 8:1:1 split
    # First 80:20 split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=random_state
    )

    # Then split 20% into 10:10 (i.e., 50:50)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=random_state
    )

    print(f"Train set: {len(train_files)} samples")
    print(f"Val set: {len(val_files)} samples")
    print(f"Test set: {len(test_files)} samples")

    # Calculate label statistics from training set
    train_labels_np = np.array(train_labels)
    label_mean = float(train_labels_np.mean())
    label_std = float(train_labels_np.std())
    print(f"Label stats (from train): mean={label_mean:.4f}, std={label_std:.4f}")

    # Create Datasets with label standardization
    train_dataset = BMDDataset(train_files, train_labels, img_size, label_mean, label_std)
    val_dataset = BMDDataset(val_files, val_labels, img_size, label_mean, label_std)
    test_dataset = BMDDataset(test_files, test_labels, img_size, label_mean, label_std)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    label_stats = {'mean': label_mean, 'std': label_std}
    return train_loader, val_loader, test_loader, label_stats


if __name__ == '__main__':
    # Test data loading
    data_dir = 'data/Sagittal_T1_FLAIR'
    xlsx_path = 'data/metadata.xlsx'

    train_loader, val_loader, test_loader, label_stats = create_dataloaders(data_dir, xlsx_path)

    # Test one batch
    for images, labels in train_loader:
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Standardized label range: {labels.min():.3f} - {labels.max():.3f}")
        # Inverse transform to original scale
        original_labels = labels * label_stats['std'] + label_stats['mean']
        print(f"Original label range: {original_labels.min():.3f} - {original_labels.max():.3f}")
        break
