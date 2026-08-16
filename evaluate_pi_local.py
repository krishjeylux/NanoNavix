#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from adaptive_matcher import adaptive_localize


PROJECT_ROOT = Path(".")


def resolve_image_path(path_string):
    """
    Resolve image paths stored in the manifest.

    Handles paths such as:
        ./final_dataset/validation/reference/00000.png
        final_dataset/validation/reference/00000.png

    without adding the split prefix a second time.
    """

    path = Path(str(path_string))

    # Remove leading "./"
    path_string = str(path)

    while path_string.startswith("./"):
        path_string = path_string[2:]

    path = PROJECT_ROOT / path_string

    return path


def evaluate(split):

    manifest_path = Path(
        f"final_dataset/{split}/manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    print("=" * 60)
    print(f"PI + LOCAL MULTI-SCALE ZNCC — {split.upper()}")
    print("=" * 60)

    results = []

    total_runtime = 0.0

    for i, row in df.iterrows():

        # -----------------------------------------------------
        # IMAGE PATHS
        # -----------------------------------------------------

        reference_path = resolve_image_path(
            row["reference_path"]
        )

        search_path = resolve_image_path(
            row["search_path"]
        )

        # -----------------------------------------------------
        # LOAD IMAGES
        # -----------------------------------------------------

        reference = cv2.imread(
            str(reference_path),
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            str(search_path),
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None:
            raise FileNotFoundError(
                f"Could not load reference image:\n"
                f"{reference_path}"
            )

        if search is None:
            raise FileNotFoundError(
                f"Could not load search image:\n"
                f"{search_path}"
            )

        # -----------------------------------------------------
        # PI + LOCAL ZNCC
        # -----------------------------------------------------

        start = time.perf_counter()

        result = adaptive_localize(
            reference,
            search,
            iterations=6,
            initial_window=100.0
        )

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        total_runtime += runtime_ms

        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        pred_x = float(result["x"])
        pred_y = float(result["y"])

        # Ground truth
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        # Euclidean localization error
        error = float(
            np.hypot(
                pred_x - gt_x,
                pred_y - gt_y
            )
        )

        # -----------------------------------------------------
        # STORE RESULT
        # -----------------------------------------------------

        results.append({
            "id": row["id"],
            "architecture": row["architecture"],

            "gt_x": gt_x,
            "gt_y": gt_y,

            "pred_x": pred_x,
            "pred_y": pred_y,

            "localization_error_px": error,

            "zncc_score": float(result["score"]),
            "scale": float(result["scale"]),
            "iterations": int(result["iterations"]),

            "runtime_ms": runtime_ms,
        })

        # -----------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    # =========================================================
    # RESULTS DATAFRAME
    # =========================================================

    results_df = pd.DataFrame(results)

    errors = results_df[
        "localization_error_px"
    ]

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Samples                  : "
        f"{len(results_df)}"
    )

    print(
        f"Mean localization error  : "
        f"{errors.mean():.3f} px"
    )

    print(
        f"Median localization error: "
        f"{errors.median():.3f} px"
    )

    print(
        f"Maximum localization error: "
        f"{errors.max():.3f} px"
    )

    print(
        f"< 1 px localization rate : "
        f"{(errors < 1.0).mean() * 100:.2f}%"
    )

    print(
        f"< 3 px localization rate : "
        f"{(errors < 3.0).mean() * 100:.2f}%"
    )

    print(
        f"< 5 px localization rate : "
        f"{(errors < 5.0).mean() * 100:.2f}%"
    )

    print(
        f"> 5 px failure rate       : "
        f"{(errors > 5.0).mean() * 100:.2f}%"
    )

    print(
        f"> 50 px failure rate      : "
        f"{(errors > 50.0).mean() * 100:.2f}%"
    )

    print(
        f"> 100 px failure rate     : "
        f"{(errors > 100.0).mean() * 100:.2f}%"
    )

    print(
        f"Mean ZNCC score           : "
        f"{results_df['zncc_score'].mean():.4f}"
    )

    print(
        f"Mean runtime              : "
        f"{results_df['runtime_ms'].mean():.3f} ms"
    )

    print(
        f"Total runtime             : "
        f"{total_runtime / 1000.0:.3f} s"
    )

    # =========================================================
    # ARCHITECTURE ANALYSIS
    # =========================================================

    print()
    print("Architecture:")

    architecture_stats = (
        results_df
        .groupby("architecture")[
            "localization_error_px"
        ]
        .agg(["count", "mean", "median"])
    )

    print(architecture_stats)

    # =========================================================
    # WORST CASES
    # =========================================================

    print()
    print("Worst 20 cases:")

    worst = (
        results_df
        .sort_values(
            "localization_error_px",
            ascending=False
        )
        .head(20)
    )

    print(
        worst.to_string(index=False)
    )

    # =========================================================
    # ERROR BUCKETS
    # =========================================================

    print()
    print("Error buckets:")

    buckets = pd.cut(
        errors,
        bins=[
            -np.inf,
            1,
            3,
            5,
            10,
            25,
            50,
            100,
            250,
            500,
            np.inf,
        ],
        labels=[
            "<=1",
            "1-3",
            "3-5",
            "5-10",
            "10-25",
            "25-50",
            "50-100",
            "100-250",
            "250-500",
            "500+",
        ],
        right=True,
    )

    print(
        buckets.value_counts(
            sort=False
        )
    )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    output_path = Path(
        f"pi_local_results_{split}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        split = sys.argv[1]
    else:
        split = "validation"

    if split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "Split must be one of: "
            "train, validation, test"
        )

    evaluate(split)