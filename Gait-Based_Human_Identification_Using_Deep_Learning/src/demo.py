import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from model import GaitModel

# --- CONFIGURATION ---
WEIGHTS_PATH = 'weights/gait_model.pth' # Make sure this matches your saved weights file!
DATA_PATH = 'data/processed'

# Let's use 5 subjects for a lightning-fast demo
DEMO_SUBJECTS = ['075', '076', '077', '078', '079'] 
PROBE_SUBJECT = '077' 
PROBE_CONDITION = 'cl-01' # Testing the difficult "Coat" condition

def load_sequence(subject, condition, angle='090'):
    """Loads a sequence, prepares it for the model, and keeps raw frames for plotting."""
    path = os.path.join(DATA_PATH, subject, condition, f"{angle}.pkl")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None, None
        
    with open(path, 'rb') as f:
        seq = pickle.load(f)
        
    # Standardize to 30 frames for the model
    if len(seq) >= 30:
        model_seq = seq[:30]
    else:
        model_seq = np.concatenate([seq, np.tile(seq[-1:], (30-len(seq), 1, 1))], axis=0)
        
    # Create tensor [1, 30, 1, 64, 64]
    tensor = torch.from_numpy(model_seq).float().unsqueeze(0).unsqueeze(2) / 255.0
    return tensor, seq

def run_demo():
    print("--- Loading Model ---")
    device = torch.device("cpu")
    model = GaitModel(num_classes=74).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print("--- Building Gallery (Normal Walks) ---")
    gallery_features = []
    gallery_raw_frames = {} # Store frames to plot later

    with torch.no_grad():
        for subj in DEMO_SUBJECTS:
            tensor, raw_frames = load_sequence(subj, 'nm-01')
            if tensor is not None:
                feat = model(tensor).squeeze().numpy()
                gallery_features.append(feat)
                gallery_raw_frames[subj] = raw_frames

        gallery_features = np.array(gallery_features)

        print(f"--- Processing Probe (Subject {PROBE_SUBJECT} in Coat) ---")
        probe_tensor, probe_raw_frames = load_sequence(PROBE_SUBJECT, PROBE_CONDITION)
        probe_feat = model(probe_tensor).squeeze().numpy()

        # Calculate Cosine Distance
        distances = cdist(probe_feat.reshape(1, -1), gallery_features, metric='cosine')[0]
        
        # Find the best match
        pred_idx = np.argmin(distances)
        pred_subject = DEMO_SUBJECTS[pred_idx]
        
        match_status = "SUCCESS!" if pred_subject == PROBE_SUBJECT else "FAILED!"
        print(f"True Identity: {PROBE_SUBJECT} | Predicted Identity: {pred_subject} -> {match_status}")

        # --- VISUALIZATION ---
        print("Generating visualization...")
        fig, axes = plt.subplots(2, 6, figsize=(16, 6))
        fig.suptitle(f"Gait Identification Demo\nTrue ID: {PROBE_SUBJECT} | Predicted ID: {pred_subject} ({match_status})", fontsize=16, fontweight='bold')

        # Plot Probe (Top Row)
        step_probe = len(probe_raw_frames) // 6
        axes[0, 0].set_ylabel(f"PROBE\n({PROBE_CONDITION})", fontsize=12, fontweight='bold', rotation=0, labelpad=40, va='center')
        for i in range(6):
            axes[0, i].imshow(probe_raw_frames[i * step_probe], cmap='gray')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

        # Plot Matched Gallery (Bottom Row)
        matched_frames = gallery_raw_frames[pred_subject]
        step_gallery = len(matched_frames) // 6
        axes[1, 0].set_ylabel(f"MATCHED\nGALLERY\n(nm-01)", fontsize=12, fontweight='bold', rotation=0, labelpad=40, va='center')
        for i in range(6):
            axes[1, i].imshow(matched_frames[i * step_gallery], cmap='gray')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

if __name__ == "__main__":
    run_demo()