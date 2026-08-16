#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from baseline_solution.siamese_dataset import SiameseCandidateDataset
from baseline_solution.siamese_verifier import SiameseVerifier

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = SiameseCandidateDataset(project_root, "train", cache_file="train_cache.pkl")
    val_dataset = SiameseCandidateDataset(project_root, "validation", cache_file="val_cache.pkl")
    
    # Pass train stats to val dataset
    val_dataset.stats = train_dataset.get_stats()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Calculate pos_weight
    num_pos = sum(1 for s in train_dataset.samples if s["label"] == 1.0)
    num_neg = len(train_dataset.samples) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32).to(device)
    print(f"Training samples: {len(train_dataset)} (Pos: {num_pos}, Neg: {num_neg})")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Calculated pos_weight: {pos_weight.item():.2f}")
    
    model = SiameseVerifier(mode="cnn_plus_classical").to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if trainable_params > 500000:
        raise ValueError("Model exceeds 500,000 trainable parameters limit!")
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for ref, search, feats, labels in train_loader:
            ref, search, feats, labels = ref.to(device), search.to(device), feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(ref, search, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ref.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for ref, search, feats, labels in val_loader:
                ref, search, feats, labels = ref.to(device), search.to(device), feats.to(device), labels.to(device)
                outputs = model(ref, search, feats)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * ref.size(0)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
                
        val_loss /= len(val_dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        val_auc = roc_auc_score(all_labels, all_preds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": train_dataset.get_stats(),
                "mode": "cnn_plus_classical",
                "epoch": epoch,
                "val_auc": val_auc
            }
            torch.save(checkpoint, "best_verifier.pth")
            print("  -> Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_auc"], label="Val AUC", color='green')
    plt.legend()
    plt.title("Validation AUC")
    
    plt.tight_layout()
    plt.savefig("verifier_training_curves.png")
    plt.close()
    
    print(f"Training completed. Best Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
