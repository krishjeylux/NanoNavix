#!/usr/bin/env python3
import argparse
import os
import time
import math
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.topk_matcher import topk_zncc_match, arbitrate_candidates

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
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("--mode", choices=["zncc_only", "zncc_psr", "zncc_psr_scale", "full", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nms-radius", type=int, default=25)
    parser.add_argument("--output-dir", default="./topk_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    manifest_path = os.path.join(project_root, "final_dataset", args.split, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    
    # Evaluate Top-K for all samples
    all_sample_results = []
    modes_to_run = ["zncc_only", "zncc_psr", "zncc_psr_scale", "full"] if args.mode == "all" else [args.mode]
    
    print(f"Evaluating Top-K on {args.split} split ({len(df)} samples)...")
    start_total = time.time()
    
    for i, row in df.iterrows():
        sample_id = int(row["id"])
        arch = row["architecture"]
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        
        ref = load_image_robustly(project_root, args.split, row["reference_path"])
        search = load_image_robustly(project_root, args.split, row["search_path"])
        
        start_time = time.time()
        candidates = topk_zncc_match(ref, search, top_k=args.top_k, nms_radius=args.nms_radius)
        end_time = time.time()
        
        # Calculate recall at K
        recall_at = {1: False, 3: False, 5: False, 10: False}
        if candidates:
            for k in [1, 3, 5, 10]:
                for c in candidates[:k]:
                    err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                    if err <= 5.0:
                        recall_at[k] = True
                        break
                        
        sample_data = {
            "id": sample_id,
            "architecture": arch,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "candidates": candidates,
            "runtime_ms": (end_time - start_time) * 1000,
            "recall_1": recall_at[1],
            "recall_3": recall_at[3],
            "recall_5": recall_at[5],
            "recall_10": recall_at[10]
        }
        all_sample_results.append(sample_data)
        
    end_total = time.time()
    print(f"Candidate generation completed. Total time: {end_total - start_total:.2f}s")
    
    # Store results per mode
    mode_metrics = {}
    
    for mode in modes_to_run:
        results = []
        for sample in all_sample_results:
            candidates = sample["candidates"]
            gt_x = sample["gt_x"]
            gt_y = sample["gt_y"]
            
            # Arbitrate
            start_arb = time.time()
            best_cand = arbitrate_candidates(candidates, mode=mode)
            end_arb = time.time()
            
            runtime = sample["runtime_ms"] + (end_arb - start_arb) * 1000
            
            err = math.hypot(best_cand["x"] - gt_x, best_cand["y"] - gt_y) if best_cand else np.inf
            
            results.append({
                "id": sample["id"],
                "architecture": sample["architecture"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "pred_x": best_cand["x"] if best_cand else -1,
                "pred_y": best_cand["y"] if best_cand else -1,
                "error_px": err,
                "runtime_ms": runtime,
                "zncc_score": best_cand["score"] if best_cand else -1,
                "psr": best_cand["psr"] if best_cand else -1,
                "cross_scale_consistency": best_cand["cross_scale_consistency"] if best_cand else -1,
                "recall_1": sample["recall_1"],
                "recall_3": sample["recall_3"],
                "recall_5": sample["recall_5"],
                "recall_10": sample["recall_10"]
            })
            
        res_df = pd.DataFrame(results)
        
        metrics = {
            "mode": mode,
            "mean_error": res_df["error_px"].mean(),
            "median_error": res_df["error_px"].median(),
            "max_error": res_df["error_px"].max(),
            "le_1": (res_df["error_px"] <= 1.0).mean() * 100,
            "le_3": (res_df["error_px"] <= 3.0).mean() * 100,
            "le_5": (res_df["error_px"] <= 5.0).mean() * 100,
            "gt_5": (res_df["error_px"] > 5.0).mean() * 100,
            "gt_50": (res_df["error_px"] > 50.0).mean() * 100,
            "gt_100": (res_df["error_px"] > 100.0).mean() * 100,
            "recall_1": res_df["recall_1"].mean() * 100,
            "recall_3": res_df["recall_3"].mean() * 100,
            "recall_5": res_df["recall_5"].mean() * 100,
            "recall_10": res_df["recall_10"].mean() * 100,
            "mean_runtime": res_df["runtime_ms"].mean(),
            "total_runtime": res_df["runtime_ms"].sum() / 1000.0,
            "df": res_df
        }
        mode_metrics[mode] = metrics
        
        # Save CSV
        csv_path = os.path.join(args.output_dir, f"topk_results_{args.split}_{mode}.csv")
        res_df.to_csv(csv_path, index=False)
        
    # Print Ablation Table
    print("\n" + "="*85)
    print("PHASE 2 ABLATION RESULTS")
    print("="*85)
    print(f"{'Mode':<15} | {'Mean Err':<8} | {'Med Err':<8} | {'<5px %':<6} | {'>50px %':<7} | {'>100px %':<8} | {'Rec@5':<5} | {'Rec@10':<6} | {'Runtime (ms)':<12}")
    print("-" * 85)
    for mode in modes_to_run:
        m = mode_metrics[mode]
        print(f"{mode:<15} | {m['mean_error']:<8.2f} | {m['median_error']:<8.2f} | {m['le_5']:<6.1f} | {m['gt_50']:<7.1f} | {m['gt_100']:<8.1f} | {m['recall_5']:<5.1f} | {m['recall_10']:<6.1f} | {m['mean_runtime']:<12.1f}")
        
    # Print diagnostics for catastrophic baseline cases
    if "zncc_only" in mode_metrics:
        baseline_df = mode_metrics["zncc_only"]["df"]
        catastrophic_indices = baseline_df[baseline_df["error_px"] > 100.0].index
        
        n_catastrophic = len(catastrophic_indices)
        
        print("\n" + "="*85)
        print("DIAGNOSTIC ANALYSIS OF CATASTROPHIC BASELINE FAILURES (>100px)")
        print("="*85)
        print(f"Baseline catastrophic cases: {n_catastrophic}")
        
        if n_catastrophic > 0:
            in_top10 = baseline_df.loc[catastrophic_indices, "recall_10"].sum()
            in_top5 = baseline_df.loc[catastrophic_indices, "recall_5"].sum()
            in_top3 = baseline_df.loc[catastrophic_indices, "recall_3"].sum()
            
            print(f"True match in Top-10: {in_top10}")
            print(f"True match in Top-5:  {in_top5}")
            print(f"True match in Top-3:  {in_top3}")
            
            for mode in modes_to_run:
                if mode == "zncc_only": continue
                mode_df = mode_metrics[mode]["df"]
                recovered = (mode_df.loc[catastrophic_indices, "error_px"] <= 5.0).sum()
                print(f"Recovered by {mode:<15}: {recovered}")
                
            print("\nInterpretation:")
            rec10 = mode_metrics['zncc_only']['recall_10']
            rem_cat = mode_metrics.get('full', mode_metrics.get(modes_to_run[-1]))['gt_100']
            if rec10 > 90:
                if rem_cat > 5.0:
                    print(f"Candidate generation is working (Recall@10 = {rec10:.1f}%), but arbitration is not perfect yet (remaining >100px = {rem_cat:.1f}%).")
                else:
                    print("Candidate generation and arbitration are both working well.")
            else:
                print(f"Candidate generation is missing some matches (Recall@10 = {rec10:.1f}%). NMS or candidate extraction may need tuning.")
                
    # Visual Diagnostics
    print("\nGenerating visual diagnostics in topk_results/ ...")
    if "zncc_only" in mode_metrics and "full" in mode_metrics:
        b_df = mode_metrics["zncc_only"]["df"]
        f_df = mode_metrics["full"]["df"]
        
        succ_idx = b_df[b_df["error_px"] <= 5.0].index
        if not succ_idx.empty:
            _plot_diagnostic(all_sample_results[succ_idx[0]], b_df.loc[succ_idx[0]], f_df.loc[succ_idx[0]], args.split, "normal_successful", project_root)
            
        cat_fix_idx = b_df[(b_df["error_px"] > 100.0) & (f_df["error_px"] <= 5.0)].index
        if not cat_fix_idx.empty:
            _plot_diagnostic(all_sample_results[cat_fix_idx[0]], b_df.loc[cat_fix_idx[0]], f_df.loc[cat_fix_idx[0]], args.split, "recovered_catastrophic", project_root)
            
        absent_idx = b_df[(b_df["error_px"] > 100.0) & (~b_df["recall_10"])].index
        if not absent_idx.empty:
            _plot_diagnostic(all_sample_results[absent_idx[0]], b_df.loc[absent_idx[0]], f_df.loc[absent_idx[0]], args.split, "absent_from_topk", project_root)
            
    print("Done.")

def _plot_diagnostic(sample_data, baseline_res, full_res, split, name, project_root):
    df = pd.read_csv(os.path.join(project_root, "final_dataset", split, "manifest.csv"))
    row = df[df["id"] == sample_data["id"]].iloc[0]
    
    search_img = load_image_robustly(project_root, split, row["search_path"])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(search_img, cmap="gray")
    
    gt_x, gt_y = sample_data["gt_x"], sample_data["gt_y"]
    plt.plot(gt_x, gt_y, 'g+', markersize=15, markeredgewidth=2, label="Ground Truth")
    
    b_x, b_y = baseline_res["pred_x"], baseline_res["pred_y"]
    plt.plot(b_x, b_y, 'rx', markersize=12, markeredgewidth=2, label="Baseline Prediction")
    
    f_x, f_y = full_res["pred_x"], full_res["pred_y"]
    plt.plot(f_x, f_y, 'b*', markersize=12, markeredgewidth=2, label="Full Arbitration Prediction")
    
    for i, c in enumerate(sample_data["candidates"]):
        plt.plot(c["x"], c["y"], 'yo', markersize=4, alpha=0.6)
        plt.text(c["x"] + 5, c["y"] + 5, str(i+1), color='yellow', fontsize=8)
        
    plt.legend()
    plt.title(f"Diagnostic: {name} (ID: {sample_data['id']})")
    
    out_dir = os.path.join(project_root, "topk_results")
    plt.savefig(os.path.join(out_dir, f"diag_{split}_{name}.png"))
    plt.close()

if __name__ == "__main__":
    main()
