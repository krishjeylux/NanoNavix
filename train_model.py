import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import DriftSenseDataset
from model import PILocalizationNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "./final_dataset"

IMAGE_SIZE = 256

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

MODEL_PATH = "pi_localization_best.pth"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠ Using CPU")


# ============================================================
# DATA LOADERS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_dataset = DriftSenseDataset(
    DATASET_ROOT,
    split="train",
    image_size=IMAGE_SIZE
)

val_dataset = DriftSenseDataset(
    DATASET_ROOT,
    split="validation",
    image_size=IMAGE_SIZE
)

print("Training samples   :", len(train_dataset))
print("Validation samples :", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = PILocalizationNet().to(DEVICE)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print(f"Parameters: {parameter_count:,}")


# ============================================================
# LOSS
# ============================================================

criterion = nn.SmoothL1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_pixel_error": []
}


best_val_error = float("inf")


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

start_time = time.time()


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        targets = batch["target"].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    pixel_errors = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            targets = batch["target"].to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            val_loss += loss.item() * images.size(0)

            # Convert normalized coordinates
            # back to 1000 x 1000 pixels.

            pred_pixels = predictions * 1000.0
            true_pixels = targets * 1000.0

            errors = torch.sqrt(
                torch.sum(
                    (pred_pixels - true_pixels) ** 2,
                    dim=1
                )
            )

            pixel_errors.extend(
                errors.detach().cpu().numpy()
            )

    val_loss /= len(val_dataset)

    mean_pixel_error = float(
        np.mean(pixel_errors)
    )

    median_pixel_error = float(
        np.median(pixel_errors)
    )

    under_1px = float(
        np.mean(
            np.array(pixel_errors) < 1.0
        ) * 100
    )

    under_5px = float(
        np.mean(
            np.array(pixel_errors) < 5.0
        ) * 100
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if mean_pixel_error < best_val_error:

        best_val_error = mean_pixel_error

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_error": mean_pixel_error,
                "val_loss": val_loss,
            },
            MODEL_PATH
        )

        best_marker = " ★ BEST"

    else:
        best_marker = ""


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_pixel_error"].append(
        mean_pixel_error
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Error: {mean_pixel_error:.2f} px | "
        f"Median: {median_pixel_error:.2f} px | "
        f"<1px: {under_1px:.1f}% | "
        f"<5px: {under_5px:.1f}% | "
        f"LR: {current_lr:.2e}"
        f"{best_marker}"
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Training time       : {elapsed:.2f} seconds"
)

print(
    f"Best validation error: "
    f"{best_val_error:.3f} px"
)

print(
    f"Best model saved to : {MODEL_PATH}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.save(
    "training_history.npy",
    history,
    allow_pickle=True
)

print(
    "Training history saved to: training_history.npy"
)