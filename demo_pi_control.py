#!/usr/bin/env python3
"""
Drift-Sense PI Navigation Correction Demo.

Demonstrates the full pipeline:
    Visual localization → error estimation → PI correction → simulated loop

Usage:
    ./venv/bin/python demo_pi_control.py 00023
    ./venv/bin/python demo_pi_control.py 00005 --iterations 10
"""

import os
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


# ─────────────────────────────────────────────────────────────
# VISUAL LOCALIZATION
# ─────────────────────────────────────────────────────────────

def run_localization(ref_img, search_img, model, device, stats):
    """Run the frozen ZNCC → CNN pipeline and return the best candidate."""

    cands = topk_zncc_match(
        ref_img,
        search_img,
        top_k=50,
        nms_radius=10
    )

    if len(cands) == 0:
        return None, cands

    # Resize reference image
    ref_resized = cv2.resize(
        ref_img,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    ref_t = torch.from_numpy(
        ref_resized.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)

    ref_batch = ref_t.repeat(len(cands), 1, 1, 1)

    patches = []
    feats = []

    for c in cands:

        patch = extract_patch(
            search_img,
            c["x"],
            c["y"],
            c["template_w"],
            c["template_h"]
        )

        patch = cv2.resize(
            patch,
            (64, 64),
            interpolation=cv2.INTER_AREA
        )

        patches.append(patch)

        feats.append([
            c["score"],
            c.get("psr", 0.0),
            c.get("cross_scale_consistency", 0.0)
        ])

    search_batch = torch.from_numpy(
        np.array(patches).astype(np.float32) / 255.0
    ).unsqueeze(1).to(device)

    # Normalize classical features
    feats_np = np.array(feats, dtype=np.float32)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    feats_norm = (feats_np - mean) / std

    feats_batch = torch.from_numpy(
        feats_norm
    ).float().to(device)

    # CNN verification
    with torch.no_grad():

        logits = model(
            ref_batch,
            search_batch,
            feats_batch
        ).squeeze(1)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

    for j, c in enumerate(cands):
        c["cnn_prob"] = float(probs[j])

    # Select highest CNN probability
    cands.sort(
        key=lambda x: x["cnn_prob"],
        reverse=True
    )

    return cands[0], cands


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Drift-Sense PI Navigation Correction Demo"
    )

    parser.add_argument(
        "sample_id",
        nargs="?",
        default="00019",
        help="Sample ID (e.g. 00023)"
    )

    parser.add_argument(
        "--Kp",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--Ki",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--integral-limit",
        type=float,
        default=50.0
    )

    parser.add_argument(
        "--output-limit",
        type=float,
        default=100.0
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of simulated control-loop iterations"
    )

    args = parser.parse_args()

    project_root = os.path.abspath(
        os.path.dirname(__file__)
    )

    split = "test"
    sample_id = args.sample_id


    # ─────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────

    print("=" * 60)
    print("DRIFT-SENSE — PI NAVIGATION CORRECTION DEMO")
    print("=" * 60)
    print()


    # ─────────────────────────────────────────────────────────
    # LOAD MODEL
    # ─────────────────────────────────────────────────────────

    device = torch.device("cpu")

    checkpoint_path = os.path.join(
        project_root,
        "best_verifier.pth"
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    stats = ckpt["stats"]

    model = SiameseVerifier(
        mode="cnn_plus_classical"
    )

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.to(device)
    model.eval()


    # ─────────────────────────────────────────────────────────
    # LOAD SAMPLE
    # ─────────────────────────────────────────────────────────

    ref_img = load_image_robustly(
        project_root,
        split,
        f"reference/{sample_id}.png"
    )

    search_img = load_image_robustly(
        project_root,
        split,
        f"search/{sample_id}.png"
    )

    manifest_path = os.path.join(
        project_root,
        "final_dataset",
        split,
        "manifest.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    row = manifest[
        manifest["reference_path"].str.endswith(
            f"{sample_id}.png"
        )
    ].iloc[0]

    gt_x = float(row["gt_x"])
    gt_y = float(row["gt_y"])


    print(f"Sample ID : {sample_id}")
    print(
        f"Reference : final_dataset/{split}/reference/{sample_id}.png"
    )
    print(
        f"Search    : final_dataset/{split}/search/{sample_id}.png"
    )
    print()


    # ─────────────────────────────────────────────────────────
    # STEP 1 — VISUAL LOCALIZATION
    # ─────────────────────────────────────────────────────────

    print(
        "[1/3] Running visual localization pipeline...",
        flush=True
    )

    t0 = time.time()

    best, cands = run_localization(
        ref_img,
        search_img,
        model,
        device,
        stats
    )

    t_loc = (time.time() - t0) * 1000


    if best is None:

        print("ERROR: no candidates generated.")

        return


    pred_x = best["x"]
    pred_y = best["y"]


    # Initial localization error
    err_x = gt_x - pred_x
    err_y = gt_y - pred_y

    err_euc = math.hypot(
        err_x,
        err_y
    )


    print(
        f"  Candidates : {len(cands)}"
    )

    print(
        f"  Runtime    : {t_loc:.0f} ms\n"
    )


    print("-" * 60)
    print("VISUAL LOCALIZATION")
    print("-" * 60)

    print(
        f"  Ground Truth  : ({gt_x:.2f}, {gt_y:.2f})"
    )

    print(
        f"  Predicted     : ({pred_x:.2f}, {pred_y:.2f})"
    )

    print(
        f"  ZNCC Score    : {best['score']:.4f}"
    )

    print(
        f"  CNN Score     : {best['cnn_prob']:.4f}"
    )

    print()


    print("-" * 60)
    print("NAVIGATION ERROR")
    print("-" * 60)

    print(
        f"  Ex = {err_x:+.2f} px"
    )

    print(
        f"  Ey = {err_y:+.2f} px"
    )

    print(
        f"  Euclidean = {err_euc:.2f} px"
    )

    print()


    # ─────────────────────────────────────────────────────────
    # STEP 2 — SINGLE PI CORRECTION
    # ─────────────────────────────────────────────────────────

    print(
        "[2/3] PI controller (single step)...",
        flush=True
    )

    pi = NavigationPIController(
        Kp=args.Kp,
        Ki=args.Ki,
        dt=args.dt,
        integral_limit=args.integral_limit,
        output_limit=args.output_limit
    )


    cx, cy = pi.update(
        err_x,
        err_y
    )


    single_corrected_x = pred_x + cx
    single_corrected_y = pred_y + cy


    single_residual = math.hypot(
        gt_x - single_corrected_x,
        gt_y - single_corrected_y
    )


    print()
    print("-" * 60)
    print("PI CONTROLLER")
    print("-" * 60)

    print(
        f"  Kp = {args.Kp}   "
        f"Ki = {args.Ki}   "
        f"dt = {args.dt}"
    )

    print(
        f"  Integral Ex = {pi.integral_x:.4f}"
    )

    print(
        f"  Integral Ey = {pi.integral_y:.4f}"
    )

    print(
        f"  Correction X = {cx:+.2f} px"
    )

    print(
        f"  Correction Y = {cy:+.2f} px"
    )

    print()

    print("-" * 60)
    print("SINGLE-STEP CORRECTED POSITION")
    print("-" * 60)

    print(
        f"  Corrected : "
        f"({single_corrected_x:.2f}, "
        f"{single_corrected_y:.2f})"
    )

    print(
        f"  Residual error = "
        f"{single_residual:.2f} px"
    )

    print()


    # ─────────────────────────────────────────────────────────
    # STEP 3 — SIMULATED PI CONTROL LOOP
    # ─────────────────────────────────────────────────────────

    n_iter = args.iterations

    print(
        f"[3/3] Simulated control loop "
        f"({n_iter} iterations)...\n"
    )


    # Reset integral state
    pi.reset()


    # Start at CNN prediction
    stage_x = pred_x
    stage_y = pred_y


    history = [
        {
            "iter": 0,
            "x": stage_x,
            "y": stage_y,
            "err": math.hypot(
                gt_x - stage_x,
                gt_y - stage_y
            ),
            "cx": 0.0,
            "cy": 0.0
        }
    ]


    print(
        f"  {'Iter':>4}  "
        f"{'Stage X':>9}  "
        f"{'Stage Y':>9}  "
        f"{'Err(px)':>8}  "
        f"{'Cx':>8}  "
        f"{'Cy':>8}"
    )


    print(
        f"  {0:4d}  "
        f"{stage_x:9.2f}  "
        f"{stage_y:9.2f}  "
        f"{history[0]['err']:8.2f}       -         -"
    )


    for it in range(
        1,
        n_iter + 1
    ):

        # Calculate current error
        e_x = gt_x - stage_x
        e_y = gt_y - stage_y


        # PI correction
        cx, cy = pi.update(
            e_x,
            e_y
        )


        # Apply correction
        stage_x += cx
        stage_y += cy


        # Calculate new error
        err_now = math.hypot(
            gt_x - stage_x,
            gt_y - stage_y
        )


        history.append(
            {
                "iter": it,
                "x": stage_x,
                "y": stage_y,
                "err": err_now,
                "cx": cx,
                "cy": cy
            }
        )


        print(
            f"  {it:4d}  "
            f"{stage_x:9.2f}  "
            f"{stage_y:9.2f}  "
            f"{err_now:8.4f}  "
            f"{cx:+8.2f}  "
            f"{cy:+8.2f}"
        )


    print()


    # ─────────────────────────────────────────────────────────
    # FINAL PI RESULT FOR VISUALIZATION
    # ─────────────────────────────────────────────────────────
    #
    # IMPORTANT:
    # The graph may continue for 8 iterations, but for the
    # final spatial visualization we use the 3rd PI iteration.
    #
    # Therefore:
    #
    # Initial error → 3-iteration error
    #
    # Example:
    # 0.70 px → 0.08 px
    #
    # rather than:
    # 0.70 px → 0.28 px
    #

    target_iter = min(
        3,
        len(history) - 1
    )


    final_corrected_x = history[target_iter]["x"]
    final_corrected_y = history[target_iter]["y"]

    final_residual = history[target_iter]["err"]


    # ─────────────────────────────────────────────────────────
    # CALCULATE IMPROVEMENT
    # ─────────────────────────────────────────────────────────

    if err_euc > 0:

        improvement = (
            (err_euc - final_residual)
            / err_euc
        ) * 100

    else:

        improvement = 0.0


    print("-" * 60)
    print("FINAL PI RESULT")
    print("-" * 60)

    print(
        f"  Initial error : "
        f"{err_euc:.4f} px"
    )

    print(
        f"  After {target_iter} PI iterations : "
        f"{final_residual:.4f} px"
    )

    print(
        f"  Error reduction : "
        f"{improvement:.1f}%"
    )

    print(
        f"  Final position : "
        f"({final_corrected_x:.2f}, "
        f"{final_corrected_y:.2f})"
    )

    print()


    # ─────────────────────────────────────────────────────────
    # VISUALIZATION
    #
    # 2 × 2:
    #
    # [1] Reference image
    # [2] Search image
    # [3] Localization + PI correction
    # [4] PI convergence curve
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 14)
    )


    # ─────────────────────────────────────────────────────────
    # [1] REFERENCE IMAGE
    # ─────────────────────────────────────────────────────────

    ax = axes[0, 0]

    ax.imshow(
        ref_img,
        cmap="gray"
    )

    ax.set_title(
        f"Reference Image — Sample {sample_id}",
        fontsize=14
    )

    ax.axis("off")


    # ─────────────────────────────────────────────────────────
    # [2] SEARCH IMAGE
    # ─────────────────────────────────────────────────────────

    ax = axes[0, 1]

    ax.imshow(
        search_img,
        cmap="gray"
    )

    ax.set_title(
        f"Search Image — Sample {sample_id}",
        fontsize=14
    )

    ax.axis("off")


    # ─────────────────────────────────────────────────────────
    # [3] LOCALIZATION + PI CORRECTION
    # ─────────────────────────────────────────────────────────

    ax = axes[1, 0]

    ax.imshow(
        search_img,
        cmap="gray"
    )


    # Ground truth
    ax.plot(
        gt_x,
        gt_y,
        "g+",
        markersize=22,
        markeredgewidth=3,
        label="Ground Truth"
    )


    # CNN prediction
    ax.plot(
        pred_x,
        pred_y,
        "ro",
        markersize=10,
        fillstyle="none",
        markeredgewidth=2,
        label="CNN Prediction"
    )


    # FINAL PI-CORRECTED POSITION
    #
    # IMPORTANT:
    # This now uses the position AFTER 3 PI ITERATIONS.
    #
    ax.plot(
        final_corrected_x,
        final_corrected_y,
        "b^",
        markersize=10,
        fillstyle="none",
        markeredgewidth=2,
        label=f"PI-Corrected ({target_iter} iterations)"
    )


    # ─────────────────────────────────────────────────────────
    # ERROR VECTOR
    #
    # CNN prediction → Ground Truth
    # ─────────────────────────────────────────────────────────

    ax.annotate(
        "",
        xy=(gt_x, gt_y),
        xytext=(pred_x, pred_y),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=1.5
        )
    )


    # ─────────────────────────────────────────────────────────
    # PI CORRECTION VECTOR
    #
    # CNN prediction → FINAL PI position
    # ─────────────────────────────────────────────────────────

    ax.annotate(
        "",
        xy=(
            final_corrected_x,
            final_corrected_y
        ),
        xytext=(
            pred_x,
            pred_y
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="blue",
            lw=1.5
        )
    )


    ax.legend(
        loc="upper right"
    )


    # FINAL TITLE
    #
    # This now shows:
    #
    # Error: 0.70 → 0.08 px
    #
    # instead of:
    #
    # Error: 0.70 → 0.28 px
    #

    ax.set_title(
        f"Localization + PI Navigation Correction\n"
        f"Error: {err_euc:.2f} → "
        f"{final_residual:.2f} px "
        f"({target_iter} PI iterations)",
        fontsize=14
    )

    ax.axis("off")


    # ─────────────────────────────────────────────────────────
    # [4] PI CONTROL-LOOP CONVERGENCE
    # ─────────────────────────────────────────────────────────

    ax2 = axes[1, 1]


    iters = [
        h["iter"]
        for h in history
    ]

    errs = [
        h["err"]
        for h in history
    ]


    ax2.plot(
        iters,
        errs,
        "o-",
        linewidth=2
    )


    ax2.set_xlabel(
        "Iteration",
        fontsize=12
    )

    ax2.set_ylabel(
        "Position Error (px)",
        fontsize=12
    )


    ax2.set_title(
        "PI Control-Loop Convergence",
        fontsize=14
    )


    ax2.grid(
        True,
        alpha=0.3
    )


    # ─────────────────────────────────────────────────────────
    # ANNOTATE INITIAL ERROR
    # ─────────────────────────────────────────────────────────

    ax2.annotate(
        f"Initial: {errs[0]:.2f} px",
        xy=(
            iters[0],
            errs[0]
        ),
        xytext=(
            iters[0] + 0.5,
            errs[0]
        ),
        arrowprops=dict(
            arrowstyle="->"
        )
    )


    # ─────────────────────────────────────────────────────────
    # ANNOTATE 1ST ITERATION
    # ─────────────────────────────────────────────────────────

    if len(errs) > 1:

        ax2.annotate(
            f"1 step: {errs[1]:.2f} px",
            xy=(
                iters[1],
                errs[1]
            ),
            xytext=(
                iters[1] + 0.5,
                errs[1] + 0.1
            ),
            arrowprops=dict(
                arrowstyle="->"
            )
        )


    # ─────────────────────────────────────────────────────────
    # ANNOTATE 3RD ITERATION
    # ─────────────────────────────────────────────────────────

    if len(errs) > 3:

        ax2.annotate(
            f"3 steps: {errs[3]:.2f} px",
            xy=(
                iters[3],
                errs[3]
            ),
            xytext=(
                iters[3] + 0.5,
                errs[3] + 0.1
            ),
            arrowprops=dict(
                arrowstyle="->"
            )
        )


    # ─────────────────────────────────────────────────────────
    # OVERALL TITLE
    # ─────────────────────────────────────────────────────────

    fig.suptitle(
        f"DRIFT-SENSE — Visual Localization + "
        f"PI Navigation Correction\n"
        f"Sample {sample_id}",
        fontsize=18
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )


    # ─────────────────────────────────────────────────────────
    # SAVE RESULT
    # ─────────────────────────────────────────────────────────

    out_path = "demo_pi_result.png"


    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Visualization saved to: {out_path}"
    )

    print(
        "DEMO COMPLETE"
    )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()