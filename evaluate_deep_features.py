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
import torch

from baseline_solution.multifeature_matcher import generate_candidates, combine_and_deduplicate
from baseline_solution.deep_feature_matcher import DeepFeatureExtractor, generate_deep_candidates

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
    out_dir = "./deep_feature_results"
    os.makedirs(out_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    manifest_path = os.path.join(project_root, "final_dataset", split, "manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Load baseline catastrophic cases from previous topk results
    baseline_res_path = os.path.join("./topk_results", "topk_results_validation_zncc_only.csv")
    catastrophic_ids = []
    if os.path.exists(baseline_res_path):
        bdf = pd.read_csv(baseline_res_path)
        catastrophic_ids = bdf[bdf["error_px"] > 100.0]["id"].tolist()
        
    print(f"Loaded {len(catastrophic_ids)} catastrophic cases from previous baseline run.")
    
    # Init feature extractors
    try:
        extractor_layer2 = DeepFeatureExtractor(layer_name="layer2")
        extractor_layer3 = DeepFeatureExtractor(layer_name="layer3")
    except Exception as e:
        print(f"Failed to initialize ResNet extractors: {e}")
        return
        
    methods = [
        "intensity", 
        "deep_layer2", 
        "deep_layer3", 
        "intensity_layer2", 
        "intensity_layer3"
    ]
    
    results = {m: {"r1":0, "r3":0, "r5":0, "r10":0, "r20":0, "r50":0, 
                   "runtime_ms": 0, "dists": []} for m in methods}
                   
    catastrophic_results = []
    all_sample_results = []
    
    diag_data = {
        "succ_zncc": None,
        "deep_recovers": None,
        "both_fail": None,
        "nominate_different": None
    }
    
    print(f"Evaluating deep feature candidate generation on {split} ({len(df)} samples)...")
    
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
        
        # 2. Deep Layer2
        t0 = time.time()
        cands_l2 = generate_deep_candidates(ref, search, extractor_layer2, top_n_per_scale=20, nms_radius=10)
        t_l2 = (time.time() - t0) * 1000
        
        # 3. Deep Layer3
        t0 = time.time()
        cands_l3 = generate_deep_candidates(ref, search, extractor_layer3, top_n_per_scale=20, nms_radius=10)
        t_l3 = (time.time() - t0) * 1000
        
        # Combinations
        t0 = time.time()
        cands_i_l2 = combine_and_deduplicate([cands_int, cands_l2], nms_radius=10)
        t_i_l2 = t_int + t_l2 + (time.time() - t0) * 1000
        
        t0 = time.time()
        cands_i_l3 = combine_and_deduplicate([cands_int, cands_l3], nms_radius=10)
        t_i_l3 = t_int + t_l3 + (time.time() - t0) * 1000
        
        pools = {
            "intensity": cands_int,
            "deep_layer2": cands_l2,
            "deep_layer3": cands_l3,
            "intensity_layer2": cands_i_l2,
            "intensity_layer3": cands_i_l3
        }
        
        runtimes = {
            "intensity": t_int,
            "deep_layer2": t_l2,
            "deep_layer3": t_l3,
            "intensity_layer2": t_i_l2,
            "intensity_layer3": t_i_l3
        }
        
        sample_res = {"id": sample_id}
        
        for m in methods:
            results[m]["runtime_ms"] += runtimes[m]
            
            matched_ranks = []
            for idx, c in enumerate(pools[m]):
                err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                if err <= 5.0:
                    matched_ranks.append(idx + 1)
                    
            best_rank = matched_ranks[0] if matched_ranks else float('inf')
            
            for k in [1, 3, 5, 10, 20, 50]:
                if best_rank <= k:
                    results[m][f"r{k}"] += 1
                    
            if len(pools[m]) > 0:
                top1_err = math.hypot(pools[m][0]["x"] - gt_x, pools[m][0]["y"] - gt_y)
                results[m]["dists"].append(top1_err)
                
            sample_res[f"{m}_in_top10"] = best_rank <= 10
            sample_res[f"{m}_in_top20"] = best_rank <= 20
            sample_res[f"{m}_in_top50"] = best_rank <= 50
            
        all_sample_results.append(sample_res)
        
        # Track catastrophic
        if sample_id in catastrophic_ids:
            cat_info = {
                "id": sample_id,
                "intensity_in_top50": sample_res["intensity_in_top50"],
                "l2_in_top10": sample_res["deep_layer2_in_top10"],
                "l2_in_top20": sample_res["deep_layer2_in_top20"],
                "l3_in_top10": sample_res["deep_layer3_in_top10"],
                "l3_in_top20": sample_res["deep_layer3_in_top20"],
                "i_l2_in_top10": sample_res["intensity_layer2_in_top10"],
                "i_l2_in_top20": sample_res["intensity_layer2_in_top20"],
                "i_l3_in_top10": sample_res["intensity_layer3_in_top10"],
                "i_l3_in_top20": sample_res["intensity_layer3_in_top20"]
            }
            catastrophic_results.append(cat_info)
            
            # Diagnostic: Deep recovers GT
            if not sample_res["intensity_in_top50"] and sample_res["deep_layer3_in_top10"]:
                if diag_data["deep_recovers"] is None:
                    diag_data["deep_recovers"] = (sample_id, pools)
                    
            # Diagnostic: Both fail
            if not sample_res["intensity_in_top50"] and not sample_res["deep_layer3_in_top50"]:
                if diag_data["both_fail"] is None:
                    diag_data["both_fail"] = (sample_id, pools)
                    
        else:
            # Diagnostic: Success ZNCC
            if sample_res["intensity_in_top10"] and sample_res["deep_layer3_in_top10"]:
                if diag_data["succ_zncc"] is None:
                    diag_data["succ_zncc"] = (sample_id, pools)
            
            # Diagnostic: Nominate different
            if len(cands_int) > 0 and len(cands_l3) > 0:
                top_int_err = math.hypot(cands_int[0]["x"] - gt_x, cands_int[0]["y"] - gt_y)
                top_l3_err = math.hypot(cands_l3[0]["x"] - gt_x, cands_l3[0]["y"] - gt_y)
                dist_between = math.hypot(cands_int[0]["x"] - cands_l3[0]["x"], cands_int[0]["y"] - cands_l3[0]["y"])
                if dist_between > 100 and top_int_err <= 5.0 and top_l3_err > 50.0:
                    if diag_data["nominate_different"] is None:
                        diag_data["nominate_different"] = (sample_id, pools)

    N = len(df)
    
    # Save CSVs
    all_res_df = pd.DataFrame(all_sample_results)
    all_res_df.to_csv("deep_feature_results_validation.csv", index=False)
    
    cat_df = pd.DataFrame(catastrophic_results)
    cat_df.to_csv(os.path.join(out_dir, "deep_feature_catastrophic_cases.csv"), index=False)
    
    summary = []
    summary.append("PHASE 2A EXPERIMENT 3 RESULTS")
    summary.append("============================================================")
    summary.append(f"DEPENDENCY STATUS:")
    summary.append(f"Torch version: {torch.__version__}")
    import torchvision
    summary.append(f"Torchvision version: {torchvision.__version__}")
    summary.append("Backbone completely frozen.")
    summary.append("Feature representations: Layer2 spatial map and Layer3 spatial map.")
    summary.append("============================================================")
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
        
        name = m.replace("_", " ").title()
        if m == "intensity": name = "Intensity ZNCC"
        if m == "deep_layer2": name = "Deep Layer2"
        if m == "deep_layer3": name = "Deep Layer3"
        
        summary.append(f"| {name:<18} | {r1:>4.1f} | {r3:>4.1f} | {r5:>4.1f} | {r10:>4.1f} | {r20:>4.1f} | {r50:>4.1f} | {rt:>4.0f} ms |")
        
    summary.append("\nDistances (Top-1 candidate):")
    for m in methods:
        mean_d = np.mean(results[m]["dists"]) if results[m]["dists"] else float('inf')
        med_d = np.median(results[m]["dists"]) if results[m]["dists"] else float('inf')
        name = m.replace("_", " ").title()
        summary.append(f"- {name}: Mean = {mean_d:.1f} px, Median = {med_d:.1f} px")
        
    summary.append("\nCatastrophic-case recovery (9 missing cases):")
    missing_9 = cat_df[~cat_df["intensity_in_top50"]]
    n_missing = len(missing_9)
    
    if n_missing > 0:
        l2_rec = missing_9["l2_in_top50"].sum() if "l2_in_top50" in missing_9 else missing_9["l2_in_top20"].sum()
        l3_rec = missing_9["l3_in_top50"].sum() if "l3_in_top50" in missing_9 else missing_9["l3_in_top20"].sum()
        
        l2_t10 = missing_9["i_l2_in_top10"].sum()
        l2_t20 = missing_9["i_l2_in_top20"].sum()
        
        l3_t10 = missing_9["i_l3_in_top10"].sum()
        l3_t20 = missing_9["i_l3_in_top20"].sum()
        
        summary.append(f"Deep layer2 recovered: {l2_t20}/{n_missing} (in Top-20)")
        summary.append(f"Deep layer3 recovered: {l3_t20}/{n_missing} (in Top-20)")
    else:
        summary.append("No missing catastrophic cases to recover in this run.")
        
    all_feat_r10 = max((results["intensity_layer2"]['r10']/N)*100, (results["intensity_layer3"]['r10']/N)*100)
    
    summary.append("\nFINAL DECISION:")
    if all_feat_r10 >= 94.0: 
        summary.append("A. Deep features provide strong complementary candidates -> proceed to candidate arbitration.")
    elif all_feat_r10 > ((results["intensity"]['r10']/N)*100) + 2.0:
        summary.append("A. Deep features provide complementary candidates -> proceed to candidate arbitration.")
    else:
        summary.append("B. Deep features do not provide meaningful complementary candidates -> STOP candidate-generation experiments and design the CNN as a patch-level match verifier instead.")
        
    with open(os.path.join(out_dir, "deep_feature_summary.txt"), "w") as f:
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
        
        for c in pools["intensity"][:10]:
            plt.plot(c["x"], c["y"], 'ro', markersize=6, alpha=0.6, label="Intensity" if c==pools["intensity"][0] else "")
            
        for c in pools["deep_layer3"][:10]:
            plt.plot(c["x"], c["y"], 'b^', markersize=6, alpha=0.6, label="Deep Layer3" if c==pools["deep_layer3"][0] else "")
            
        plt.legend()
        plt.title(f"Diagnostic: {diag_name} (ID: {sample_id})")
        plt.savefig(os.path.join(out_dir, f"diag_deep_{diag_name}.png"))
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
