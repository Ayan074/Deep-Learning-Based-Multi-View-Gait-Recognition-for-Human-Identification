import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GaitModel
from dataset import GaitDataset


# =====================================================
# CONFIGURATION
# =====================================================

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50

DATA_PATH = "data/processed"
WEIGHTS_PATH = "weights/gait_model_best.pth"

# Auto device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# VALIDATION FUNCTION
# =====================================================

def validate(model, classifier, val_loader, criterion, device):
    """
    Validation after each epoch
    Returns:
    - validation loss
    - validation accuracy
    """

    model.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            embeddings = model(frames)
            outputs = classifier(embeddings)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train():
    print(f"\nTraining on: {DEVICE}\n")

    # =================================================
    # DATASET LOADING
    # =================================================

    print("Loading Training Dataset...")
    train_dataset = GaitDataset(
        DATA_PATH,
        training=True,
        validation=False
    )

    print("\nLoading Validation Dataset...")
    val_dataset = GaitDataset(
        DATA_PATH,
        training=True,
        validation=True
    )

    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    print(f"\nTrain Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}\n")

    # =================================================
    # MODEL + CLASSIFIER
    # =================================================

    # IMPORTANT:
    # Training split is now 64 subjects
    model = GaitModel(num_classes=74).to(DEVICE)
    classifier = nn.Linear(256, 74).to(DEVICE)

    # =================================================
    # LOSS FUNCTION
    # =================================================

    # Label smoothing helps prevent overfitting
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # =================================================
    # OPTIMIZER
    # =================================================

    optimizer = optim.Adam(
        list(model.parameters()) +
        list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4   # helps reduce overfitting
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    # =================================================
    # TRAINING LOOP
    # =================================================

    best_val_acc = 0.0

    os.makedirs("weights", exist_ok=True)

    print("Starting Training...\n")

    for epoch in range(EPOCHS):

        model.train()
        classifier.train()

        running_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{EPOCHS}]",
            leave=False
        )

        for frames, labels in loop:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            embeddings = model(frames)
            outputs = classifier(embeddings)

            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(
                train_loss=f"{loss.item():.4f}"
            )

        avg_train_loss = running_loss / len(train_loader)

        # =============================================
        # VALIDATION STEP
        # =============================================

        val_loss, val_acc = validate(
            model,
            classifier,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss : {avg_train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")
        print(f"LR         : {current_lr:.6f}")

        # =============================================
        # SAVE BEST MODEL
        # =============================================

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1
            }, WEIGHTS_PATH)

            print(f"*** New Best Model Saved! "
                  f"Validation Accuracy: {best_val_acc:.2f}% ***")

        print("-" * 50)

    print("\nTraining Finished Successfully!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved at: {WEIGHTS_PATH}")



if __name__ == "__main__":
    train()