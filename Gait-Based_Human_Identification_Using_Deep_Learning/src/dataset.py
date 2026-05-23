import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GaitDataset(Dataset):
    def __init__(
        self,
        data_path,
        training=True,
        frame_count=30,
        validation=False,
        val_split=0.2
    ):

        self.data_path = data_path
        self.frame_count = frame_count
        self.samples = []

        # Safe numeric sorting
        all_subjects = sorted(
            os.listdir(data_path),
            key=lambda x: int(x)
        )

        # -----------------------------------
        # Subject Split
        # -----------------------------------

        if training:
            subjects = all_subjects[:74]   # training identities
        else:
            subjects = all_subjects[74:]   # unseen test identities

        # Fixed label mapping for training identities
        self.label_map = {
            subj: i for i, subj in enumerate(subjects)
        }

        # -----------------------------------
        # Collect all samples
        # -----------------------------------

        all_samples = []

        for subj in subjects:
            subj_path = os.path.join(data_path, subj)

            if not os.path.isdir(subj_path):
                continue

            for cond in os.listdir(subj_path):
                cond_path = os.path.join(subj_path, cond)

                if not os.path.isdir(cond_path):
                    continue

                for angle_file in os.listdir(cond_path):
                    if angle_file.endswith(".pkl"):
                        all_samples.append({
                            "path": os.path.join(cond_path, angle_file),
                            "label": self.label_map[subj]
                        })

        # -----------------------------------
        # Sample-based Train / Validation Split
        # -----------------------------------

        if training:
            np.random.seed(42)
            np.random.shuffle(all_samples)

            split_idx = int(len(all_samples) * (1 - val_split))

            if validation:
                self.samples = all_samples[split_idx:]   # last 20%
            else:
                self.samples = all_samples[:split_idx]   # first 80%

        else:
            self.samples = all_samples

        print(
            f"Loaded {len(self.samples)} samples "
            f"from {len(subjects)} subjects."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Load .pkl sequence
        with open(sample_info["path"], "rb") as f:
            sequence = pickle.load(f)

        # Shape:
        # [Total_Frames, 64, 64]

        total_frames = sequence.shape[0]

        # -----------------------------------
        # Random frame selection
        # -----------------------------------

        if total_frames >= self.frame_count:
            start = np.random.randint(
                0,
                total_frames - self.frame_count + 1
            )

            sequence = sequence[
                start:start + self.frame_count
            ]

        else:
            # Padding for short sequences
            padding = np.tile(
                sequence[-1:],
                (self.frame_count - total_frames, 1, 1)
            )

            sequence = np.concatenate(
                [sequence, padding],
                axis=0
            )

        # Convert to tensor
        sequence = (
            torch.from_numpy(sequence)
            .float()
            .unsqueeze(1) / 255.0
        )

        label = torch.tensor(
            sample_info["label"]
        ).long()

        return sequence, label


if __name__ == "__main__":
    try:
        print("Testing Training Dataset...\n")

        train_dataset = GaitDataset(
            data_path="data/processed",
            training=True,
            validation=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )

        frames, labels = next(iter(train_loader))

        print("Dataset Test Successful!")
        print(f"Batch Shape: {frames.shape}")
        print("Expected: [4, 30, 1, 64, 64]")
        print(f"Labels: {labels}")

        print("\nTesting Validation Dataset...\n")

        val_dataset = GaitDataset(
            data_path="data/processed",
            training=True,
            validation=True
        )

        print(f"Validation Samples: {len(val_dataset)}")

    except Exception as e:
        print(f"Dataset Test Note: {e}")
        print("This test will pass once .pkl files exist.")