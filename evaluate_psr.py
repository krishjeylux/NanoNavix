#!/usr/bin/env python3
import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.psr import zncc_match_with_psr

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./psr_results")
    args = parser.parse_args()
    
    # Set seed for reproducible evaluation if random ops are used
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    manifest_path = os.path.join(project_root, "final_dataset", args.split, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    
    results = []
    
    print(f"Evaluating {args.split} split ({len(df)} samples)...")
    
    for i, row in df.iterrows():
        sample_id = int(row["id"])
        arch = row["architecture"]
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        
        ref = load_image_robustly(project_root, args.split, row["reference_path"])
        search = load_image_robustly(project_root, args.split, row["search_path"])
        
        start_time = time.time()
        best_match, _ = zncc_match_with_psr(ref, search, guard_radius=10)
        end_time = time.time()
        
        runtime_ms = (end_time - start_time) * 1000
        
        pred_x = best_match["x"]
        pred_y = best_match["y"]
        error_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        
        res = {
            "id": sample_id,
            "architecture": arch,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "localization_error_px": error_px,
            "zncc_score": best_match["score"],
            "psr": best_match["psr"],
            "scale": best_match["scale"],
            "runtime_ms": runtime_ms,
            "second_peak_score": best_match.get("second_peak_score", np.nan),
            "peak_to_second_peak_difference": best_match.get("peak_diff", np.nan),
            "err_le_1": int(error_px <= 1.0),
            "err_le_3": int(error_px <= 3.0),
            "err_le_5": int(error_px <= 5.0),
            "err_gt_50": int(error_px > 50.0),
            "err_gt_100": int(error_px > 100.0)
        }
        results.append(res)
        
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"psr_results_{args.split}.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Analysis
    correct_df = res_df[res_df["localization_error_px"] <= 5.0]
    catastrophic_df = res_df[res_df["localization_error_px"] > 100.0]
    
    print("\n" + "="*60)
    print("PSR ANALYSIS")
    print("="*60)
    
    if not correct_df.empty:
        print(f"Correct matches (<=5 px) [N={len(correct_df)}]:")
        print(f"Mean PSR: {correct_df['psr'].mean():.4f}")
        print(f"Median PSR: {correct_df['psr'].median():.4f}")
        print(f"Min PSR: {correct_df['psr'].min():.4f}")
        print(f"Max PSR: {correct_df['psr'].max():.4f}")
        print(f"Std PSR: {correct_df['psr'].std():.4f}")
    else:
        print("Correct matches: 0")
        
    print()
    if not catastrophic_df.empty:
        print(f"Catastrophic matches (>100 px) [N={len(catastrophic_df)}]:")
        print(f"Mean PSR: {catastrophic_df['psr'].mean():.4f}")
        print(f"Median PSR: {catastrophic_df['psr'].median():.4f}")
        print(f"Min PSR: {catastrophic_df['psr'].min():.4f}")
        print(f"Max PSR: {catastrophic_df['psr'].max():.4f}")
        print(f"Std PSR: {catastrophic_df['psr'].std():.4f}")
    else:
        print("Catastrophic matches: 0")
        
    # Threshold sweep
    thresholds = np.linspace(res_df["psr"].min(), res_df["psr"].max(), 50)
    threshold_results = []
    
    for t in thresholds:
        accepted = res_df[res_df["psr"] >= t]
        rejected = res_df[res_df["psr"] < t]
        
        n_accepted = len(accepted)
        n_rejected = len(rejected)
        
        if n_accepted > 0:
            precision = len(accepted[accepted["localization_error_px"] <= 5.0]) / n_accepted
            catastrophic_rate = len(accepted[accepted["localization_error_px"] > 100.0]) / n_accepted
        else:
            precision = 0.0
            catastrophic_rate = 0.0
            
        recall = len(accepted[accepted["localization_error_px"] <= 5.0]) / max(len(correct_df), 1)
        
        threshold_results.append({
            "psr_threshold": t,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "precision": precision,
            "recall": recall,
            "catastrophic_rate": catastrophic_rate
        })
        
    thresh_df = pd.DataFrame(threshold_results)
    thresh_csv_path = os.path.join(args.output_dir, f"psr_threshold_analysis_{args.split}.csv")
    thresh_df.to_csv(thresh_csv_path, index=False)
    print(f"Threshold analysis saved to {thresh_csv_path}")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(res_df["psr"], res_df["localization_error_px"], alpha=0.5, s=20)
    plt.xlabel("PSR")
    plt.ylabel("Localization Error (px)")
    plt.title(f"PSR vs Localization Error ({args.split})")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, f"psr_vs_localization_error.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.hist(correct_df["psr"], bins=20, alpha=0.5, label="Correct (<=5px)")
    plt.hist(catastrophic_df["psr"], bins=20, alpha=0.5, label="Catastrophic (>100px)")
    plt.xlabel("PSR")
    plt.ylabel("Count")
    plt.title(f"PSR Distribution ({args.split})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, f"psr_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    # Correct condition is True, Incorrect is False.
    # We want True (1) to be blue and False (0) to be red. coolwarm_r maps 1 to blue and 0 to red.
    plt.scatter(res_df["zncc_score"], res_df["psr"], c=res_df["localization_error_px"] <= 5.0, cmap="coolwarm_r", alpha=0.7)
    plt.xlabel("ZNCC Score")
    plt.ylabel("PSR")
    plt.title(f"ZNCC Score vs PSR ({args.split})")
    plt.grid(True, alpha=0.3)
    
    import matplotlib.lines as mlines
    # Use standard matplotlib colors 'blue' and 'red' to match the colormap loosely
    blue_star = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Correct (<=5px)')
    red_star = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Incorrect (>5px)')
    plt.legend(handles=[blue_star, red_star])
    plt.savefig(os.path.join(args.output_dir, f"zncc_vs_psr.png"))
    plt.close()
    print("Plots saved.")

    print("\nConclusion:")
    if not correct_df.empty and not catastrophic_df.empty:
        diff = correct_df["psr"].mean() - catastrophic_df["psr"].mean()
        if diff > 1.0:
            print(f"PSR appears to be a strong signal for distinguishing correct from catastrophic matches. On average, correct matches have a PSR {diff:.2f} higher than catastrophic matches.")
        elif diff > 0.5:
            print(f"PSR provides a moderate signal for distinguishing correct from catastrophic matches. On average, correct matches have a PSR {diff:.2f} higher than catastrophic matches.")
        else:
            print("PSR does not appear to provide a strong separation between correct and catastrophic matches based on the mean difference.")
    else:
        print("Insufficient data to draw a conclusion on PSR separation.")

if __name__ == "__main__":
    main()
