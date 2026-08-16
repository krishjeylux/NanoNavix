#!/usr/bin/env python3
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.topk_matcher import topk_zncc_match, arbitrate_candidates
from baseline_solution.siamese_verifier import SiameseVerifier
from baseline_solution.siamese_dataset import load_image_robustly, extract_patch

def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    if len(sys.argv) > 1:
        split = sys.argv[1]
    else:
        split = "validation"
        
    out_dir = f"./verifier_results_{split}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path = "best_verifier.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stats = checkpoint["stats"]
    
    model = SiameseVerifier(mode="cnn_plus_classical")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    manifest_path = os.path.join(project_root, "final_dataset", split, "manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Load baseline catastrophic cases
    baseline_res_path = os.path.join("./topk_results", f"topk_results_{split}_zncc_only.csv")
    catastrophic_ids = []
    if os.path.exists(baseline_res_path):
        bdf = pd.read_csv(baseline_res_path)
        catastrophic_ids = bdf[bdf["error_px"] > 100.0]["id"].tolist()
        
    methods = ["zncc_top1", "classical_arbitration", "cnn_only", "cnn_plus_classical"]
    metrics = {m: {"errors": [], "r1": 0, "r3": 0, "r5": 0, "runtime_ms": 0} for m in methods}
    oracle_recall = {10: 0, 20: 0, 50: 0}
    catastrophic_results = []
    all_sample_results = []
    
    diag_data = {
        "normal_success": None,
        "catastrophic_recovers_cnn": None,
        "catastrophic_fails_cnn": None,
        "gt_absent": None,
        "false_high_zncc": None,
        "correct_arbitration": None
    }
    
    print(f"Evaluating verifier on {split} set ({len(df)} samples)...")
    
    for i, row in df.iterrows():
        sample_id = int(row["id"])
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        
        ref = load_image_robustly(project_root, split, row["reference_path"])
        search = load_image_robustly(project_root, split, row["search_path"])
        
        t0 = time.time()
        cands = topk_zncc_match(ref, search, top_k=50, nms_radius=10)
        
        # Oracle analysis
        gt_ranks = []
        for idx, c in enumerate(cands):
            err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
            if err <= 5.0:
                gt_ranks.append(idx + 1)
                
        best_rank = gt_ranks[0] if gt_ranks else float('inf')
        
        if best_rank <= 10: oracle_recall[10] += 1
        if best_rank <= 20: oracle_recall[20] += 1
        if best_rank <= 50: oracle_recall[50] += 1
        
        if len(cands) == 0:
            continue
            
        # ZNCC Top-1
        zncc_top1 = cands[0]
        
        # Classical arbitration (ZNCC + PSR + cross_scale)
        classical_top1 = arbitrate_candidates(cands, mode="full", weights={"zncc": 1.0, "psr": 1.0, "consistency": 1.0})
        
        # Evaluate CNN on all candidates
        ref_resized = cv2.resize(ref, (64, 64), interpolation=cv2.INTER_AREA)
        ref_tensor = torch.from_numpy(ref_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        ref_batch = ref_tensor.repeat(len(cands), 1, 1, 1)
        
        search_patches = []
        classical_feats = []
        
        for c in cands:
            patch = extract_patch(search, c["x"], c["y"], c["template_w"], c["template_h"])
            patch_resized = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
            search_patches.append(patch_resized)
            
            f = [c["score"], c.get("psr", 0.0), c.get("cross_scale_consistency", 0.0)]
            classical_feats.append(f)
            
        search_batch = torch.from_numpy(np.array(search_patches).astype(np.float32) / 255.0).unsqueeze(1).to(device)
        
        # Normalize classical features
        feats_np = np.array(classical_feats, dtype=np.float32)
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        feats_norm = (feats_np - mean) / std
        feats_batch = torch.from_numpy(feats_norm).float().to(device)
        
        with torch.no_grad():
            # CNN + Classical
            combined_logits = model(ref_batch, search_batch, feats_batch).squeeze(1)
            combined_probs = torch.sigmoid(combined_logits).cpu().numpy()
            
            # CNN Only (by zeroing classical features)
            zeros_batch = torch.zeros_like(feats_batch)
            cnn_logits = model(ref_batch, search_batch, zeros_batch).squeeze(1)
            cnn_probs = torch.sigmoid(cnn_logits).cpu().numpy()
            
        t_total = (time.time() - t0) * 1000
        
        # Rank candidates
        cands_cnn = []
        cands_combined = []
        for j, c in enumerate(cands):
            c_cnn = c.copy()
            c_cnn["cnn_prob"] = float(cnn_probs[j])
            cands_cnn.append(c_cnn)
            
            c_comb = c.copy()
            c_comb["combined_prob"] = float(combined_probs[j])
            cands_combined.append(c_comb)
            
        cands_cnn.sort(key=lambda x: x["cnn_prob"], reverse=True)
        cands_combined.sort(key=lambda x: x["combined_prob"], reverse=True)
        
        # Results
        preds = {
            "zncc_top1": [zncc_top1],
            "classical_arbitration": [classical_top1] if classical_top1 else cands,
            "cnn_only": cands_cnn,
            "cnn_plus_classical": cands_combined
        }
        
        sample_res = {"id": sample_id, "gt_x": gt_x, "gt_y": gt_y}
        
        for m in methods:
            metrics[m]["runtime_ms"] += t_total
            
            # Recalls
            m_ranks = []
            for idx, c in enumerate(preds[m]):
                err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                if err <= 5.0:
                    m_ranks.append(idx + 1)
                    
            b_rank = m_ranks[0] if m_ranks else float('inf')
            
            if b_rank <= 1: metrics[m]["r1"] += 1
            if b_rank <= 3: metrics[m]["r3"] += 1
            if b_rank <= 5: metrics[m]["r5"] += 1
            
            # Top-1 Error
            if len(preds[m]) > 0:
                top1_c = preds[m][0]
                top1_err = math.hypot(top1_c["x"] - gt_x, top1_c["y"] - gt_y)
                metrics[m]["errors"].append(top1_err)
                sample_res[f"{m}_err"] = top1_err
            else:
                metrics[m]["errors"].append(float('inf'))
                sample_res[f"{m}_err"] = float('inf')
                
        all_sample_results.append(sample_res)
            
        # Catastrophic tracking
        if sample_id in catastrophic_ids:
            gt_in_top50 = best_rank <= 50
            
            zncc_err = math.hypot(zncc_top1["x"] - gt_x, zncc_top1["y"] - gt_y)
            class_err = math.hypot(classical_top1["x"] - gt_x, classical_top1["y"] - gt_y)
            cnn_err = math.hypot(cands_cnn[0]["x"] - gt_x, cands_cnn[0]["y"] - gt_y)
            comb_err = math.hypot(cands_combined[0]["x"] - gt_x, cands_combined[0]["y"] - gt_y)
            
            catastrophic_results.append({
                "id": sample_id,
                "gt_in_top50": gt_in_top50,
                "zncc_recovers": zncc_err <= 5.0,
                "classical_recovers": class_err <= 5.0,
                "cnn_recovers": cnn_err <= 5.0,
                "combined_recovers": comb_err <= 5.0
            })
            
            if not gt_in_top50:
                if diag_data["gt_absent"] is None:
                    diag_data["gt_absent"] = (sample_id, cands, preds)
            else:
                if comb_err <= 5.0:
                    if diag_data["catastrophic_recovers_cnn"] is None:
                        diag_data["catastrophic_recovers_cnn"] = (sample_id, cands, preds)
                else:
                    if diag_data["catastrophic_fails_cnn"] is None:
                        diag_data["catastrophic_fails_cnn"] = (sample_id, cands, preds)
        else:
            top1_err = math.hypot(zncc_top1["x"] - gt_x, zncc_top1["y"] - gt_y)
            if top1_err <= 5.0:
                if diag_data["normal_success"] is None:
                    diag_data["normal_success"] = (sample_id, cands, preds)
                    
            if top1_err > 50.0 and zncc_top1["score"] > 0.9:
                if diag_data["false_high_zncc"] is None:
                    diag_data["false_high_zncc"] = (sample_id, cands, preds)

    N = len(df)
    
    res_df = pd.DataFrame(all_sample_results)
    res_df.to_csv(f"verifier_results_{split}.csv", index=False)
    
    summary = []
    summary.append("PHASE 3: PATCH-LEVEL CNN VERIFIER RESULTS")
    summary.append("==================================================")
    
    # Oracle
    summary.append(f"ZNCC Top-K Candidate Ceiling (Oracle):")
    summary.append(f"  Top-10: {oracle_recall[10]/N*100:.1f}%")
    summary.append(f"  Top-20: {oracle_recall[20]/N*100:.1f}%")
    summary.append(f"  Top-50: {oracle_recall[50]/N*100:.1f}%")
    summary.append("")
    
    # Metrics
    summary.append("| Mode | Mean Err | Med Err | <5px | >50px | >100px | R@1 | R@3 | R@5 | Runtime |")
    summary.append("|------|----------|---------|------|-------|--------|-----|-----|-----|---------|")
    
    for m in methods:
        errs = np.array(metrics[m]["errors"])
        mean_err = errs.mean()
        med_err = np.median(errs)
        lt5 = (errs <= 5.0).sum() / N * 100
        gt50 = (errs > 50.0).sum() / N * 100
        gt100 = (errs > 100.0).sum() / N * 100
        
        r1 = metrics[m]["r1"] / N * 100
        r3 = metrics[m]["r3"] / N * 100
        r5 = metrics[m]["r5"] / N * 100
        rt = metrics[m]["runtime_ms"] / N
        
        name = m.replace("_", " ").title()
        summary.append(f"| {name[:12]} | {mean_err:8.1f} | {med_err:7.1f} | {lt5:4.1f} | {gt50:5.1f} | {gt100:6.1f} | {r1:3.0f} | {r3:3.0f} | {r5:3.0f} | {rt:6.0f} |")
        
    summary.append("\nCatastrophic Failure Analysis (from 23 baseline cases):")
    if catastrophic_results:
        cat_df = pd.DataFrame(catastrophic_results)
        gt_absent = cat_df[~cat_df["gt_in_top50"]]
        gt_present = cat_df[cat_df["gt_in_top50"]]
        
        summary.append(f"  UNRECOVERABLE (GT absent from Top-50): {len(gt_absent)}")
        
        if len(gt_present) > 0:
            summary.append(f"  Recoverable cases (GT in Top-50): {len(gt_present)}")
            summary.append(f"    - CNN recovered: {gt_present['cnn_recovers'].sum()}")
            summary.append(f"    - Classical recovered: {gt_present['classical_recovers'].sum()}")
            summary.append(f"    - CNN+Classical recovered: {gt_present['combined_recovers'].sum()}")
    
    with open(os.path.join(out_dir, f"verifier_summary_{split}.txt"), "w") as f:
        f.write("\n".join(summary))
        
    print("\n".join(summary))

    # Diagnostics
    print("\nGenerating diagnostic visualizations...")
    for name, data in diag_data.items():
        if data is None: continue
        sample_id, cands, preds = data
        
        row = df[df["id"] == sample_id].iloc[0]
        search = load_image_robustly(project_root, split, row["search_path"])
        
        plt.figure(figsize=(12, 12))
        plt.imshow(search, cmap="gray")
        plt.plot(row["gt_x"], row["gt_y"], 'g+', markersize=20, markeredgewidth=3, label="Ground Truth")
        
        zncc_c = preds["zncc_top1"][0]
        plt.plot(zncc_c["x"], zncc_c["y"], 'ro', markersize=10, fillstyle='none', label="ZNCC Top-1")
        
        comb_c = preds["cnn_plus_classical"][0]
        plt.plot(comb_c["x"], comb_c["y"], 'b^', markersize=10, fillstyle='none', label="CNN+Classical")
        
        plt.legend()
        plt.title(f"{name} (ID: {sample_id})")
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close()
        
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
