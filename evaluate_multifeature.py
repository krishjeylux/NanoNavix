#!/usr/bin/env python3
import os
import sys
import time
import math
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.multifeature_matcher import generate_candidates, combine_and_deduplicate

def load_image_robustly(project_root, split, image_path):
    if os.path.isabs(image_path):
        path = image_path
    elif image_path.startswith("./final_dataset") or image_path.startswith("final_dataset"):
        if image_path.startswith("./"):
            path = os.path.join(project_root, image_path[2:])
        else:
            path = os.path.join(project_root, image_path)
    else:
        path = os.path.join(project_root, "final_dataset", split, image_path)
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def main():
    split = "validation"
    out_dir = "./topk_results"
    os.makedirs(out_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    manifest_path = os.path.join(project_root, "final_dataset", split, "manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Load baseline catastrophic cases
    baseline_res_path = os.path.join(out_dir, "topk_results_validation_zncc_only.csv")
    catastrophic_ids = []
    if os.path.exists(baseline_res_path):
        bdf = pd.read_csv(baseline_res_path)
        catastrophic_ids = bdf[bdf["error_px"] > 100.0]["id"].tolist()
    
    methods = [
        "intensity", 
        "edge", 
        "highpass", 
        "intensity_edge", 
        "intensity_highpass", 
        "all_three"
    ]
    
    # Data structures for tracking
    results = {m: {"r1":0, "r3":0, "r5":0, "r10":0, "r20":0, "r50":0, "runtime_ms": 0} for m in methods}
    
    catastrophic_results = []
    all_sample_results = []
    
    # For diagnostic plots
    diag_data = {
        "int_fail_edge_succ": None,
        "int_fail_hp_succ": None,
        "all_fail": None,
        "all_succ": None
    }
    
    print(f"Evaluating multifeature candidate generation on {split} ({len(df)} samples)...")
    
    for i, row in df.iterrows():
        sample_id = int(row["id"])
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        
        ref = load_image_robustly(project_root, split, row["reference_path"])
        search = load_image_robustly(project_root, split, row["search_path"])
        
        # 1. Intensity
        t0 = time.time()
        cands_int = generate_candidates(ref, search, "intensity", top_n_per_scale=20, nms_radius=10)
        t_int = (time.time() - t0) * 1000
        
        # 2. Edge
        t0 = time.time()
        cands_edge = generate_candidates(ref, search, "edge", top_n_per_scale=20, nms_radius=10)
        t_edge = (time.time() - t0) * 1000
        
        # 3. High-pass
        t0 = time.time()
        cands_hp = generate_candidates(ref, search, "highpass", top_n_per_scale=20, nms_radius=10)
        t_hp = (time.time() - t0) * 1000
        
        # Combinations
        t0 = time.time()
        cands_ie = combine_and_deduplicate([cands_int, cands_edge], nms_radius=10)
        t_ie = t_int + t_edge + (time.time() - t0) * 1000
        
        t0 = time.time()
        cands_ih = combine_and_deduplicate([cands_int, cands_hp], nms_radius=10)
        t_ih = t_int + t_hp + (time.time() - t0) * 1000
        
        t0 = time.time()
        cands_all = combine_and_deduplicate([cands_int, cands_edge, cands_hp], nms_radius=10)
        t_all = t_int + t_edge + t_hp + (time.time() - t0) * 1000
        
        pools = {
            "intensity": cands_int,
            "edge": cands_edge,
            "highpass": cands_hp,
            "intensity_edge": cands_ie,
            "intensity_highpass": cands_ih,
            "all_three": cands_all
        }
        
        runtimes = {
            "intensity": t_int,
            "edge": t_edge,
            "highpass": t_hp,
            "intensity_edge": t_ie,
            "intensity_highpass": t_ih,
            "all_three": t_all
        }
        
        sample_res = {"id": sample_id}
        
        for m in methods:
            results[m]["runtime_ms"] += runtimes[m]
            
            # Find best rank of true match
            matched_ranks = []
            for idx, c in enumerate(pools[m]):
                err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                if err <= 5.0:
                    matched_ranks.append(idx + 1)
            
            best_rank = matched_ranks[0] if matched_ranks else float('inf')
            
            for k in [1, 3, 5, 10, 20, 50]:
                if best_rank <= k:
                    results[m][f"r{k}"] += 1
                    
            # For saving to CSV
            sample_res[f"{m}_best_rank"] = best_rank if best_rank != float('inf') else -1
            sample_res[f"{m}_in_top50"] = best_rank <= 50
            sample_res[f"{m}_in_top10"] = best_rank <= 10
            
        all_sample_results.append(sample_res)
        
        # Track catastrophic
        if sample_id in catastrophic_ids:
            cat_info = {
                "id": sample_id,
                "intensity_contains_gt": sample_res["intensity_in_top50"],
                "edge_contains_gt": sample_res["edge_in_top50"],
                "highpass_contains_gt": sample_res["highpass_in_top50"],
                "intensity_edge_contains_gt": sample_res["intensity_edge_in_top50"],
                "intensity_highpass_contains_gt": sample_res["intensity_highpass_in_top50"],
                "all_three_contains_gt": sample_res["all_three_in_top50"],
                "recovered_by": "none"
            }
            if not cat_info["intensity_contains_gt"]:
                if cat_info["edge_contains_gt"] and not cat_info["highpass_contains_gt"]:
                    cat_info["recovered_by"] = "edge"
                elif cat_info["highpass_contains_gt"] and not cat_info["edge_contains_gt"]:
                    cat_info["recovered_by"] = "highpass"
                elif cat_info["edge_contains_gt"] and cat_info["highpass_contains_gt"]:
                    cat_info["recovered_by"] = "both"
            catastrophic_results.append(cat_info)
            
        # Diagnostic cases
        if sample_res["intensity_in_top10"] and sample_res["edge_in_top10"] and sample_res["highpass_in_top10"]:
            if diag_data["all_succ"] is None:
                diag_data["all_succ"] = (sample_id, pools)
        if not sample_res["intensity_in_top50"] and sample_res["edge_in_top50"]:
            if diag_data["int_fail_edge_succ"] is None:
                diag_data["int_fail_edge_succ"] = (sample_id, pools)
        if not sample_res["intensity_in_top50"] and sample_res["highpass_in_top50"]:
            if diag_data["int_fail_hp_succ"] is None:
                diag_data["int_fail_hp_succ"] = (sample_id, pools)
        if not sample_res["intensity_in_top50"] and not sample_res["edge_in_top50"] and not sample_res["highpass_in_top50"]:
            if diag_data["all_fail"] is None:
                diag_data["all_fail"] = (sample_id, pools)

    N = len(df)
    
    # Save CSVs
    all_res_df = pd.DataFrame(all_sample_results)
    all_res_df.to_csv(os.path.join(out_dir, "multiffeature_candidate_validation.csv"), index=False)
    
    cat_df = pd.DataFrame(catastrophic_results)
    cat_df.to_csv(os.path.join(out_dir, "multiffeature_catastrophic_cases.csv"), index=False)
    
    # Summary
    summary = []
    summary.append("PHASE 2A EXPERIMENT 2 RESULTS")
    summary.append("")
    summary.append("| Method | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | Runtime |")
    summary.append("|--------|-----|-----|-----|------|------|------|---------|")
    
    for m in methods:
        r1 = (results[m]['r1']/N)*100
        r3 = (results[m]['r3']/N)*100
        r5 = (results[m]['r5']/N)*100
        r10 = (results[m]['r10']/N)*100
        r20 = (results[m]['r20']/N)*100
        r50 = (results[m]['r50']/N)*100
        rt = results[m]['runtime_ms']/N
        
        # Proper table formatting
        method_name = m.replace("_", " ").title() if "_" in m else m.title()
        if m == "all_three": method_name = "All Three"
        if m == "intensity_edge": method_name = "Intensity + Edge"
        if m == "intensity_highpass": method_name = "Intensity + High-pass"
        
        summary.append(f"| {method_name:<21} | {r1:>4.1f}% | {r3:>4.1f}% | {r5:>4.1f}% | {r10:>5.1f}% | {r20:>5.1f}% | {r50:>5.1f}% | {rt:>5.1f}ms |")
        
    summary.append("\nCatastrophic-case recovery:")
    if not cat_df.empty:
        int_only = cat_df["intensity_contains_gt"].sum()
        edge_adds = cat_df["recovered_by"].isin(["edge", "both"]).sum()
        hp_adds = cat_df["recovered_by"].isin(["highpass", "both"]).sum()
        all_feat = cat_df["all_three_contains_gt"].sum()
        total_cat = len(cat_df)
        
        summary.append(f"- Intensity only: {int_only}/{total_cat}")
        summary.append(f"- Edge adds: {edge_adds} new cases")
        summary.append(f"- High-pass adds: {hp_adds} new cases")
        summary.append(f"- All features: {all_feat}/{total_cat}")
    else:
        summary.append("- No catastrophic cases found.")
        
    all_feat_r10 = (results["all_three"]['r10']/N)*100
    
    summary.append("\nCANDIDATE GENERATION STATUS:")
    if all_feat_r10 >= 94.0: 
        summary.append("READY FOR ARBITRATION")
    else:
        summary.append("NOT READY")
        
    with open(os.path.join(out_dir, "multiffeature_summary.txt"), "w") as f:
        f.write("\n".join(summary))
        
    print("\n".join(summary))
    
    # Visualizations
    print("\nGenerating visual diagnostics...")
    for diag_name, data in diag_data.items():
        if data is None:
            continue
        sample_id, pools = data
        row = df[df["id"] == sample_id].iloc[0]
        
        search_img = load_image_robustly(project_root, split, row["search_path"])
        plt.figure(figsize=(12, 12))
        plt.imshow(search_img, cmap="gray")
        
        plt.plot(row["gt_x"], row["gt_y"], 'g+', markersize=20, markeredgewidth=3, label="Ground Truth")
        
        # Plot top 10 from intensity
        for c in pools["intensity"][:10]:
            plt.plot(c["x"], c["y"], 'ro', markersize=6, alpha=0.6, label="Intensity" if c==pools["intensity"][0] else "")
            
        for c in pools["edge"][:10]:
            plt.plot(c["x"], c["y"], 'b^', markersize=6, alpha=0.6, label="Edge" if c==pools["edge"][0] else "")
            
        for c in pools["highpass"][:10]:
            plt.plot(c["x"], c["y"], 'ys', markersize=6, alpha=0.6, label="High-pass" if c==pools["highpass"][0] else "")
            
        plt.legend()
        plt.title(f"Diagnostic: {diag_name} (ID: {sample_id})")
        plt.savefig(os.path.join(out_dir, f"diag_multifeature_{diag_name}.png"))
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
