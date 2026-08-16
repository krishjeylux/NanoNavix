#!/usr/bin/env python3
"""
Drift-Sense PI Navigation Correction Demo.

Demonstrates the full pipeline:
    Visual localization  →  error estimation  →  PI correction  →  simulated loop

Usage:
    ./venv/bin/python demo_pi_control.py 00023
    ./venv/bin/python demo_pi_control.py 00005 --iterations 10
"""
import os
import sys
import time
import math
import argparse

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
from pi_controller import NavigationPIController


# ── helpers ──────────────────────────────────────────────────────────

def run_localization(ref_img, search_img, model, device, stats):
    """Run the frozen ZNCC → CNN pipeline and return the best candidate."""
    cands = topk_zncc_match(ref_img, search_img, top_k=50, nms_radius=10)
    if len(cands) == 0:
        return None, cands

    ref_resized = cv2.resize(ref_img, (64, 64), interpolation=cv2.INTER_AREA)
    ref_t = torch.from_numpy(
        ref_resized.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)
    ref_batch = ref_t.repeat(len(cands), 1, 1, 1)

    patches, feats = [], []
    for c in cands:
        p = extract_patch(search_img, c["x"], c["y"],
                          c["template_w"], c["template_h"])
        patches.append(cv2.resize(p, (64, 64), interpolation=cv2.INTER_AREA))
        feats.append([c["score"],
                      c.get("psr", 0.0),
                      c.get("cross_scale_consistency", 0.0)])

    s_batch = torch.from_numpy(
        np.array(patches).astype(np.float32) / 255.0
    ).unsqueeze(1).to(device)

    feats_np = np.array(feats, dtype=np.float32)
    mean = np.array(stats["mean"])
    std  = np.array(stats["std"])
    f_batch = torch.from_numpy((feats_np - mean) / std).float().to(device)

    with torch.no_grad():
        logits = model(ref_batch, s_batch, f_batch).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    for j, c in enumerate(cands):
        c["cnn_prob"] = float(probs[j])

    cands.sort(key=lambda x: x["cnn_prob"], reverse=True)
    return cands[0], cands


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense PI Navigation Correction Demo")
    parser.add_argument("sample_id", nargs="?", default="00019",
                        help="Sample ID (e.g. 00023)")
    parser.add_argument("--Kp", type=float, default=0.5)
    parser.add_argument("--Ki", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--integral-limit", type=float, default=50.0)
    parser.add_argument("--output-limit", type=float, default=100.0)
    parser.add_argument("--iterations", type=int, default=8,
                        help="Number of simulated control-loop iterations")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.dirname(__file__))
    split = "validation"
    sample_id = args.sample_id

    # ── header ───────────────────────────────────────────────────────
    print("=" * 60)
    print("DRIFT-SENSE — PI NAVIGATION CORRECTION DEMO")
    print("=" * 60)
    print()

    # ── load model (once) ────────────────────────────────────────────
    device = torch.device("cpu")
    ckpt = torch.load(os.path.join(project_root, "best_verifier.pth"),
                      map_location=device)
    stats = ckpt["stats"]
    model = SiameseVerifier(mode="cnn_plus_classical")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # ── load sample ──────────────────────────────────────────────────
    ref_img = load_image_robustly(project_root, split,
                                  f"reference/{sample_id}.png")
    search_img = load_image_robustly(project_root, split,
                                     f"search/{sample_id}.png")

    manifest = pd.read_csv(
        os.path.join(project_root, "final_dataset", split, "manifest.csv"))
    row = manifest[manifest["reference_path"].str.endswith(
        f"{sample_id}.png")].iloc[0]
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

    print(f"Sample ID : {sample_id}")
    print(f"Reference : final_dataset/{split}/reference/{sample_id}.png")
    print(f"Search    : final_dataset/{split}/search/{sample_id}.png")
    print()

    # ── single-shot localization ─────────────────────────────────────
    print("[1/3] Running visual localization pipeline...", flush=True)
    t0 = time.time()
    best, cands = run_localization(ref_img, search_img, model, device, stats)
    t_loc = (time.time() - t0) * 1000

    if best is None:
        print("  ERROR: no candidates generated.")
        return

    pred_x, pred_y = best["x"], best["y"]
    err_x = gt_x - pred_x
    err_y = gt_y - pred_y
    err_euc = math.hypot(err_x, err_y)

    print(f"  Candidates : {len(cands)}")
    print(f"  Runtime    : {t_loc:.0f} ms\n")

    print("-" * 60)
    print("VISUAL LOCALIZATION")
    print("-" * 60)
    print(f"  Ground Truth  : ({gt_x:.2f}, {gt_y:.2f})")
    print(f"  Predicted     : ({pred_x:.2f}, {pred_y:.2f})")
    print(f"  ZNCC Score    : {best['score']:.4f}")
    print(f"  CNN Score     : {best['cnn_prob']:.4f}")
    print()

    print("-" * 60)
    print("NAVIGATION ERROR")
    print("-" * 60)
    print(f"  Ex = {err_x:+.2f} px")
    print(f"  Ey = {err_y:+.2f} px")
    print(f"  Euclidean = {err_euc:.2f} px")
    print()

    # ── PI single step ───────────────────────────────────────────────
    print("[2/3] PI controller (single step)...", flush=True)
    pi = NavigationPIController(
        Kp=args.Kp, Ki=args.Ki, dt=args.dt,
        integral_limit=args.integral_limit,
        output_limit=args.output_limit)

    cx, cy = pi.update(err_x, err_y)
    corrected_x = pred_x + cx
    corrected_y = pred_y + cy
    residual = math.hypot(gt_x - corrected_x, gt_y - corrected_y)

    print()
    print("-" * 60)
    print("PI CONTROLLER")
    print("-" * 60)
    print(f"  Kp = {args.Kp}   Ki = {args.Ki}   dt = {args.dt}")
    print(f"  Integral Ex = {pi.integral_x:.4f}")
    print(f"  Integral Ey = {pi.integral_y:.4f}")
    print(f"  Correction X = {cx:+.2f} px")
    print(f"  Correction Y = {cy:+.2f} px")
    print()
    print("-" * 60)
    print("CORRECTED POSITION")
    print("-" * 60)
    print(f"  Corrected : ({corrected_x:.2f}, {corrected_y:.2f})")
    print(f"  Residual error = {residual:.2f} px")
    print()

    # ── multi-iteration simulation ───────────────────────────────────
    n_iter = args.iterations
    print(f"[3/3] Simulated control loop ({n_iter} iterations)...\n")

    pi.reset()
    # Start the simulated stage at the CNN prediction
    stage_x, stage_y = pred_x, pred_y
    history = [{"iter": 0, "x": stage_x, "y": stage_y,
                "err": math.hypot(gt_x - stage_x, gt_y - stage_y),
                "cx": 0.0, "cy": 0.0}]

    print(f"  {'Iter':>4}  {'Stage X':>9}  {'Stage Y':>9}  "
          f"{'Err(px)':>8}  {'Cx':>8}  {'Cy':>8}")
    print(f"  {0:4d}  {stage_x:9.2f}  {stage_y:9.2f}  "
          f"{history[0]['err']:8.2f}       -         -")

    for it in range(1, n_iter + 1):
        e_x = gt_x - stage_x
        e_y = gt_y - stage_y
        cx, cy = pi.update(e_x, e_y)

        # Simulate applying the correction to the stage
        stage_x += cx
        stage_y += cy
        err_now = math.hypot(gt_x - stage_x, gt_y - stage_y)

        history.append({"iter": it, "x": stage_x, "y": stage_y,
                        "err": err_now, "cx": cx, "cy": cy})
        print(f"  {it:4d}  {stage_x:9.2f}  {stage_y:9.2f}  "
              f"{err_now:8.4f}  {cx:+8.2f}  {cy:+8.2f}")

    print()

    # ── visualisation ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: spatial
    ax = axes[0]
    ax.imshow(search_img, cmap="gray")
    ax.plot(gt_x, gt_y, "g+", markersize=22, markeredgewidth=3,
            label="Ground Truth")
    ax.plot(pred_x, pred_y, "ro", markersize=10, fillstyle="none",
            markeredgewidth=2, label="CNN Prediction")
    ax.plot(corrected_x, corrected_y, "b^", markersize=10, fillstyle="none",
            markeredgewidth=2, label="PI-Corrected")

    # error vector
    ax.annotate("", xy=(gt_x, gt_y), xytext=(pred_x, pred_y),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    # correction vector
    ax.annotate("", xy=(corrected_x, corrected_y),
                xytext=(pred_x, pred_y),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    ax.legend(loc="upper right")
    ax.set_title(f"Sample {sample_id}  |  Error {err_euc:.2f} → "
                 f"{residual:.2f} px")
    ax.axis("off")

    # right: convergence
    ax2 = axes[1]
    iters = [h["iter"] for h in history]
    errs  = [h["err"] for h in history]
    ax2.plot(iters, errs, "o-", color="#2563eb", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Position Error (px)")
    ax2.set_title("PI Control-Loop Convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "demo_pi_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {out_path}")
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
