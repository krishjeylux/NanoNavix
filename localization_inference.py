#!/usr/bin/env python3
"""
Localization Inference Script
Usage:
    python localization_inference.py --reference <path> --search <path>
"""
import os
import argparse
import torch
import cv2
import numpy as np

from baseline_solution.topk_matcher import topk_zncc_match
from baseline_solution.siamese_verifier import SiameseVerifier
from baseline_solution.siamese_dataset import extract_patch

def parse_args():
    parser = argparse.ArgumentParser(description="Run NanoNavix Localization Inference")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference image not found: {args.reference}")
    if not os.path.exists(args.search):
        raise FileNotFoundError(f"Search image not found: {args.search}")
        
    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    
    # Run ZNCC
    cands = topk_zncc_match(ref_img, search_img, top_k=50, nms_radius=10)
    
    if len(cands) == 0:
        print("No candidates found.")
        return
        
    # Load Model
    device = torch.device("cpu")
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(project_root, "best_verifier.pth")
    
    if not os.path.exists(checkpoint_path):
        # Fallback to ZNCC top 1 if no model
        best_cand = cands[0]
        print(f"Predicted Center: ({best_cand['x']:.2f}, {best_cand['y']:.2f})")
        return
        
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
        
    cands.sort(key=lambda x: x["cnn_prob"], reverse=True)
    best_cand = cands[0]
    
    print(f"Predicted Center: ({best_cand['x']:.2f}, {best_cand['y']:.2f})")

if __name__ == "__main__":
    main()
