#!/usr/bin/env python3
import os
import sys
import time
import math
import pandas as pd
import numpy as np
import cv2

from baseline_solution.topk_matcher import topk_zncc_match

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
    catastrophic_ids = set()
    if os.path.exists(baseline_res_path):
        bdf = pd.read_csv(baseline_res_path)
        catastrophic_ids = set(bdf[bdf["error_px"] > 100.0]["id"].tolist())
    
    top_n_list = [20, 50, 100]
    nms_rad_list = [5, 10, 15, 20, 30]
    
    results = []
    
    print(f"Starting sweep over {len(top_n_list) * len(nms_rad_list)} configurations...")
    
    for n in top_n_list:
        for r in nms_rad_list:
            print(f"Evaluating top_n_per_scale={n}, nms_radius={r}...")
            
            start_total = time.time()
            
            recall_counts = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
            catastrophic_recoveries = {10: 0, 20: 0, 50: 0}
            
            total_pre_nms = 0
            total_post_nms = 0
            
            for i, row in df.iterrows():
                sample_id = int(row["id"])
                gt_x = float(row["gt_x"])
                gt_y = float(row["gt_y"])
                
                ref = load_image_robustly(project_root, split, row["reference_path"])
                search = load_image_robustly(project_root, split, row["search_path"])
                
                cands, stats = topk_zncc_match(
                    ref, search, 
                    top_k=50, 
                    nms_radius=r, 
                    top_n_per_scale=n, 
                    return_stats=True
                )
                
                total_pre_nms += stats["pre_nms_candidates"]
                total_post_nms += stats["post_nms_candidates"]
                
                # Evaluate recalls
                matched_ranks = []
                for idx, c in enumerate(cands):
                    err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                    if err <= 5.0:
                        matched_ranks.append(idx + 1)
                
                best_rank = matched_ranks[0] if matched_ranks else float('inf')
                
                for k in [1, 3, 5, 10, 20, 50]:
                    if best_rank <= k:
                        recall_counts[k] += 1
                        
                if sample_id in catastrophic_ids:
                    for k in [10, 20, 50]:
                        if best_rank <= k:
                            catastrophic_recoveries[k] += 1
                            
            end_total = time.time()
            total_runtime_s = end_total - start_total
            runtime_per_sample_ms = (total_runtime_s / len(df)) * 1000
            
            res = {
                "top_n_per_scale": n,
                "nms_radius": r,
                "recall_1": (recall_counts[1] / len(df)) * 100,
                "recall_3": (recall_counts[3] / len(df)) * 100,
                "recall_5": (recall_counts[5] / len(df)) * 100,
                "recall_10": (recall_counts[10] / len(df)) * 100,
                "recall_20": (recall_counts[20] / len(df)) * 100,
                "recall_50": (recall_counts[50] / len(df)) * 100,
                "catastrophic_in_top10": catastrophic_recoveries[10],
                "catastrophic_in_top20": catastrophic_recoveries[20],
                "catastrophic_in_top50": catastrophic_recoveries[50],
                "runtime_per_sample_ms": runtime_per_sample_ms,
                "total_runtime_s": total_runtime_s,
                "mean_pre_nms": total_pre_nms / len(df),
                "mean_post_nms": total_post_nms / len(df)
            }
            results.append(res)
            
    res_df = pd.DataFrame(results)
    
    # Sort as requested
    res_df = res_df.sort_values(by=["recall_10", "recall_20", "runtime_per_sample_ms"], ascending=[False, False, True])
    
    csv_path = os.path.join(out_dir, "candidate_generation_sweep_validation.csv")
    res_df.to_csv(csv_path, index=False)
    
    # Summary
    best = res_df.iloc[0]
    
    summary = []
    summary.append("CANDIDATE GENERATION SWEEP SUMMARY")
    summary.append("="*50)
    summary.append(f"Best Configuration: top_n_per_scale = {best['top_n_per_scale']}, nms_radius = {best['nms_radius']}")
    summary.append(f"Recall@1:  {best['recall_1']:.1f}%")
    summary.append(f"Recall@3:  {best['recall_3']:.1f}%")
    summary.append(f"Recall@5:  {best['recall_5']:.1f}%")
    summary.append(f"Recall@10: {best['recall_10']:.1f}%")
    summary.append(f"Recall@20: {best['recall_20']:.1f}%")
    summary.append(f"Recall@50: {best['recall_50']:.1f}%")
    summary.append(f"Catastrophic baseline cases (total {len(catastrophic_ids)}):")
    summary.append(f"  True match in Top-10: {best['catastrophic_in_top10']}")
    summary.append(f"  True match in Top-20: {best['catastrophic_in_top20']}")
    summary.append(f"  True match in Top-50: {best['catastrophic_in_top50']}")
    summary.append(f"Runtime per sample: {best['runtime_per_sample_ms']:.1f} ms")
    summary.append(f"Mean candidates before NMS: {best['mean_pre_nms']:.1f}")
    summary.append(f"Mean candidates after NMS:  {best['mean_post_nms']:.1f}")
    
    # Check if increasing top_n produces duplicates
    subset = res_df[res_df['nms_radius'] == 10]
    if len(subset) >= 2:
        small_n = subset[subset['top_n_per_scale'] == 20].iloc[0]
        large_n = subset[subset['top_n_per_scale'] == 100].iloc[0]
        diff_pre = large_n['mean_pre_nms'] - small_n['mean_pre_nms']
        diff_post = large_n['mean_post_nms'] - small_n['mean_post_nms']
        
        summary.append("\nDUPLICATE ANALYSIS:")
        summary.append(f"Increasing top_n from 20 to 100 (at nms=10) added {diff_pre:.1f} pre-NMS candidates.")
        summary.append(f"But only added {diff_post:.1f} post-NMS candidates.")
        if diff_pre > 0 and diff_post / diff_pre < 0.2:
            summary.append("Conclusion: Increasing top_n primarily produces spatial duplicates around the same physical locations, which NMS successfully removes.")
        else:
            summary.append("Conclusion: Increasing top_n produces many distinct spatial candidates.")
            
    summary_path = os.path.join(out_dir, "candidate_generation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
        
    print(f"Sweep complete. Best config: top_n={best['top_n_per_scale']}, nms_rad={best['nms_radius']} (Recall@10={best['recall_10']}%)")

if __name__ == "__main__":
    main()
