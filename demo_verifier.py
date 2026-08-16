#!/usr/bin/env python3
import os
import sys
import time
import math
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.topk_matcher import topk_zncc_match
from baseline_solution.siamese_verifier import SiameseVerifier
from baseline_solution.siamese_dataset import extract_patch, load_image_robustly

def main():
    print("============================================================")
    print("DRIFT-SENSE — NAVIGATION ERROR RECOVERY DEMO")
    print("============================================================\n")

    project_root = os.path.abspath(os.path.dirname(__file__))
    split = "validation"
    sample_id = sys.argv[1] if len(sys.argv) > 1 else "00019"

    ref_rel = f"reference/{sample_id}.png"
    search_rel = f"search/{sample_id}.png"
    
    print(f"Reference : final_dataset/{split}/{ref_rel}")
    print(f"Search    : final_dataset/{split}/{search_rel}\n")

    # [1/5] Load images and ground truth
    print("[1/5] Loading reference and search images...", end="", flush=True)
    try:
        ref_img = load_image_robustly(project_root, split, ref_rel)
        search_img = load_image_robustly(project_root, split, search_rel)
        
        manifest_path = os.path.join(project_root, "final_dataset", split, "manifest.csv")
        df = pd.read_csv(manifest_path)
        # Find row where reference_path ends with 00000.png
        row = df[df["reference_path"].str.endswith(f"{sample_id}.png")].iloc[0]
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
    except Exception as e:
        print(f" FAILED\nError: {e}")
        return
    print("      DONE")

    # [2/5] ZNCC Candidate Generation
    print("[2/5] Multi-scale ZNCC candidate generation...", end="", flush=True)
    t0 = time.time()
    cands = topk_zncc_match(ref_img, search_img, top_k=50, nms_radius=10)
    t_zncc = (time.time() - t0) * 1000
    print("   DONE")
    print(f"      Candidates generated : {len(cands)}")

    if len(cands) == 0:
        print("Error: No candidates generated.")
        return

    # [3/5] Siamese CNN verification
    print("[3/5] Siamese CNN verification...", end="", flush=True)
    t0 = time.time()
    device = torch.device("cpu")
    checkpoint_path = os.path.join(project_root, "best_verifier.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stats = checkpoint["stats"]
    
    model = SiameseVerifier(mode="cnn_plus_classical")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    ref_resized = cv2.resize(ref_img, (64, 64), interpolation=cv2.INTER_AREA)
    ref_tensor = torch.from_numpy(ref_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    ref_batch = ref_tensor.repeat(len(cands), 1, 1, 1)
    
    search_patches = []
    classical_feats = []
    for c in cands:
        patch = extract_patch(search_img, c["x"], c["y"], c["template_w"], c["template_h"])
        patch_resized = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
        search_patches.append(patch_resized)
        f = [c["score"], c.get("psr", 0.0), c.get("cross_scale_consistency", 0.0)]
        classical_feats.append(f)
        
    search_batch = torch.from_numpy(np.array(search_patches).astype(np.float32) / 255.0).unsqueeze(1).to(device)
    
    feats_np = np.array(classical_feats, dtype=np.float32)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    feats_norm = (feats_np - mean) / std
    feats_batch = torch.from_numpy(feats_norm).float().to(device)
    
    with torch.no_grad():
        combined_logits = model(ref_batch, search_batch, feats_batch).squeeze(1)
        combined_probs = torch.sigmoid(combined_logits).cpu().numpy()
        
    for j, c in enumerate(cands):
        c["cnn_prob"] = float(combined_probs[j])
        
    t_cnn = (time.time() - t0) * 1000
    print("                DONE")
    print(f"      Candidates verified  : {len(cands)}")

    # [4/5] Final candidate selection
    print("[4/5] Final candidate selection...", end="", flush=True)
    cands.sort(key=lambda x: x["cnn_prob"], reverse=True)
    best_cand = cands[0]
    print("               DONE\n")
    
    err = math.hypot(best_cand["x"] - gt_x, best_cand["y"] - gt_y)
    total_time = t_zncc + t_cnn

    print("------------------------------------------------------------")
    print("RESULT")
    print("------------------------------------------------------------")
    print(f"Ground Truth X : {gt_x:.2f}")
    print(f"Ground Truth Y : {gt_y:.2f}")
    print(f"Predicted X    : {best_cand['x']:.2f}")
    print(f"Predicted Y    : {best_cand['y']:.2f}")
    print(f"Localization Error : {err:.2f} px")
    print(f"ZNCC Score         : {best_cand['score']:.4f}")
    print(f"CNN Score          : {best_cand['cnn_prob']:.4f}")
    print(f"Runtime            : {total_time:.1f} ms")
    print("------------------------------------------------------------\n")
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(search_img, cmap="gray")
    plt.plot(gt_x, gt_y, 'g+', markersize=20, markeredgewidth=3, label="Ground Truth")
    plt.plot(best_cand["x"], best_cand["y"], 'ro', markersize=12, fillstyle='none', markeredgewidth=2, label="CNN Prediction")
    plt.title(f"Localization Demo (Error: {err:.2f} px)")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("demo_result.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample ID : {sample_id}")
    print("Visualization saved to: demo_result.png")
    print("DEMO COMPLETE")

if __name__ == "__main__":
    main()
