import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/processed"

def show_sample(subject=None, condition=None, angle=None):
    subjects = sorted(os.listdir(DATA_PATH))

    # Select subject
    if subject is None:
        subject = subjects[0]
    print(f"Selected Subject: {subject}")

    subject_path = os.path.join(DATA_PATH, subject)

    conditions = sorted(os.listdir(subject_path))

    # Select condition
    if condition is None:
        condition = conditions[0]
    print(f"Selected Condition: {condition}")

    condition_path = os.path.join(subject_path, condition)

    # Select .pkl file (angle)
    pkl_files = [f for f in os.listdir(condition_path) if f.endswith(".pkl")]

    if len(pkl_files) == 0:
        print("No .pkl file found!")
        return

    if angle is None:
        pkl_file = pkl_files[0]
    else:
        pkl_file = f"{angle}.pkl"

    print(f"Selected File: {pkl_file}")

    pkl_path = os.path.join(condition_path, pkl_file)

    # Load sequence
    with open(pkl_path, "rb") as f:
        sequence = pickle.load(f)

    print(f"Loaded Shape: {sequence.shape}")

    # Show frames
    plt.figure(figsize=(10, 10))

    for i in range(min(9, len(sequence))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(sequence[i], cmap="gray")
        plt.title(f"Frame {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Examples:
    show_sample(subject="001", condition="bg-01", angle="000")