import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from model import GaitModel


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/processed"
WEIGHTS_PATH = "weights/gait_model_best.pth"

# Test subjects (unseen subjects)
TEST_SUBJECTS = [f"{i:03d}" for i in range(75, 125)]

# Standard CASIA-B all view angles
ALL_ANGLES = [
    "000", "018", "036", "054", "072",
    "090",
    "108", "126", "144", "162", "180"
]

# Fixed gallery view
GALLERY_ANGLE = "090"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def get_feature(model, subject, condition, angle):
    """
    Extract 256-D gait embedding from one sequence
    """

    path = os.path.join(
        DATA_PATH,
        subject,
        condition,
        f"{angle}.pkl"
    )

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        sequence = pickle.load(f)

    # -----------------------------------
    # Standardize to 30 frames
    # -----------------------------------

    total_frames = len(sequence)

    if total_frames >= 30:
        sequence = sequence[:30]
    else:
        padding = np.tile(
            sequence[-1:],
            (30 - total_frames, 1, 1)
        )

        sequence = np.concatenate(
            [sequence, padding],
            axis=0
        )

    # Shape:
    # [1, 30, 1, 64, 64]

    inp = (
        torch.from_numpy(sequence)
        .float()
        .unsqueeze(0)
        .unsqueeze(2) / 255.0
    ).to(DEVICE)

    with torch.no_grad():
        embedding = model(inp)

    return embedding.squeeze().cpu().numpy()


# =====================================================
# MULTI-VIEW EVALUATION
# =====================================================

def run_multiview_evaluation():
    print("\n======================================")
    print("MULTI-VIEW GAIT EVALUATION")
    print("Gallery = 090°")
    print("Probe = ALL ANGLES")
    print("Rank-1 + Rank-5 Accuracy")
    print("======================================\n")

    print(f"Running on: {DEVICE}\n")

    # -----------------------------------
    # Load best trained model
    # -----------------------------------

    model = GaitModel(num_classes=74).to(DEVICE)

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Best model loaded successfully.")
    print(
        f"Best Validation Accuracy: "
        f"{checkpoint['best_val_acc']:.2f}%"
    )
    print(
        f"Saved from Epoch: "
        f"{checkpoint['epoch']}\n"
    )

    # =================================================
    # STEP 1: BUILD GALLERY
    # Gallery = NM-01 @ 090°
    # =================================================

    print("Building Gallery (NM-01 @ 090°)...")

    gallery_features = []
    gallery_labels = []

    for subject in tqdm(TEST_SUBJECTS):

        feature = get_feature(
            model,
            subject,
            condition="nm-01",
            angle=GALLERY_ANGLE
        )

        if feature is not None:
            gallery_features.append(feature)
            gallery_labels.append(subject)

    gallery_features = np.array(gallery_features)

    print(f"\nGallery Built Successfully")
    print(f"Gallery Size: {len(gallery_labels)} subjects\n")

    # =================================================
    # STEP 2: TEST ALL ANGLES
    # Probe = NM-02 @ All Angles
    # =================================================

    final_results = {}

    for angle in ALL_ANGLES:

        print("======================================")
        print(f"Testing Probe Angle: {angle}°")
        print("======================================")

        correct_rank1 = 0
        correct_rank5 = 0
        total = 0

        for subject in tqdm(TEST_SUBJECTS):

            probe_feature = get_feature(
                model,
                subject,
                condition="nm-02",
                angle=angle
            )

            if probe_feature is None:
                continue

            # -----------------------------------
            # Cosine Distance Matching
            # -----------------------------------

            distances = cdist(
                probe_feature.reshape(1, -1),
                gallery_features,
                metric="cosine"
            )[0]

            # -----------------------------------
            # Rank-1
            # -----------------------------------

            pred_idx = np.argmin(distances)
            predicted_subject = gallery_labels[pred_idx]

            if predicted_subject == subject:
                correct_rank1 += 1

            # -----------------------------------
            # Rank-5
            # -----------------------------------

            top5_indices = np.argsort(distances)[:5]
            top5_subjects = [
                gallery_labels[i]
                for i in top5_indices
            ]

            if subject in top5_subjects:
                correct_rank5 += 1

            total += 1

        # -----------------------------------
        # Accuracy Calculation
        # -----------------------------------

        rank1_acc = (
            (correct_rank1 / total) * 100
            if total > 0 else 0
        )

        rank5_acc = (
            (correct_rank5 / total) * 100
            if total > 0 else 0
        )

        final_results[angle] = (
            rank1_acc,
            rank5_acc
        )

        print(f"\nRank-1 Correct : {correct_rank1}")
        print(f"Rank-5 Correct : {correct_rank5}")
        print(f"Total Probes   : {total}")
        print(f"Rank-1 Accuracy: {rank1_acc:.2f}%")
        print(f"Rank-5 Accuracy: {rank5_acc:.2f}%\n")

    # =================================================
    # FINAL SUMMARY
    # =================================================

    print("\n======================================")
    print("FINAL MULTI-VIEW RESULTS SUMMARY")
    print("======================================\n")

    all_rank1 = []
    all_rank5 = []

    for angle, (r1, r5) in final_results.items():

        print(
            f"{angle}° : "
            f"Rank-1 = {r1:.2f}% | "
            f"Rank-5 = {r5:.2f}%"
        )

        all_rank1.append(r1)
        all_rank5.append(r5)

    print("\n--------------------------------------")
    print(
        f"Average Rank-1 Accuracy : "
        f"{np.mean(all_rank1):.2f}%"
    )
    print(
        f"Average Rank-5 Accuracy : "
        f"{np.mean(all_rank5):.2f}%"
    )
    print("--------------------------------------\n")

    print("Multi-View Evaluation Complete Successfully!")



if __name__ == "__main__":
    run_multiview_evaluation()