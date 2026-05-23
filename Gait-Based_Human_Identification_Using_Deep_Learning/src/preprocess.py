import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# Configuration
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
IMG_SIZE = 64


def image_preprocessor(img_path):
    """
    Preprocess a single silhouette image:
    1. Read grayscale image
    2. Detect person silhouette
    3. Crop bounding box
    4. Preserve aspect ratio while resizing
    5. Apply zero padding to keep final size 64x64
    """

    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Find all non-zero pixels (white silhouette)
    y_indices, x_indices = np.where(img > 0)

    # Skip empty/invalid frames
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    # Bounding box of the silhouette
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Crop tightly around the person
    cropped = img[y_min:y_max + 1, x_min:x_max + 1]

    # -------------------------------
    # IMPORTANT IMPROVEMENT:
    # Preserve Aspect Ratio
    # -------------------------------

    h, w = cropped.shape

    # Scale based on larger dimension
    scale = IMG_SIZE / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # Prevent zero-dimension issue
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Resize while preserving proportions
    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Create black background canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Center image on canvas
    y_offset = (IMG_SIZE - new_h) // 2
    x_offset = (IMG_SIZE - new_w) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


def run_preprocessing():
    """
    Process all raw gait images and save them
    as .pkl sequences for fast training.
    """

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Safer numeric sorting for subject folders
    subjects = sorted(
        os.listdir(RAW_DATA_PATH),
        key=lambda x: int(x)
    )

    for subj in tqdm(subjects, desc="Processing Subjects"):

        subj_path = os.path.join(RAW_DATA_PATH, subj)

        if not os.path.isdir(subj_path):
            continue

        for condition in os.listdir(subj_path):

            cond_path = os.path.join(subj_path, condition)

            if not os.path.isdir(cond_path):
                continue

            for angle in os.listdir(cond_path):

                angle_path = os.path.join(cond_path, angle)

                if not os.path.isdir(angle_path):
                    continue

                frames = []

                # Sort image frames correctly
                img_files = sorted(os.listdir(angle_path))

                for img_file in img_files:

                    img_path = os.path.join(angle_path, img_file)

                    processed_img = image_preprocessor(img_path)

                    if processed_img is not None:
                        frames.append(processed_img)

                # Save only valid sequences
                if len(frames) > 0:

                    out_dir = os.path.join(
                        PROCESSED_DATA_PATH,
                        subj,
                        condition
                    )

                    os.makedirs(out_dir, exist_ok=True)

                    out_file = os.path.join(
                        out_dir,
                        f"{angle}.pkl"
                    )

                    with open(out_file, 'wb') as f:
                        pickle.dump(np.array(frames), f)

    print("\nPreprocessing Complete Successfully!")


if __name__ == "__main__":
    print("Starting Preprocessing...")
    print("This may take a while depending on CPU/GPU and dataset size.\n")

    run_preprocessing()