import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from adaptive_matcher import adaptive_match


def evaluate(split="validation"):
    manifest_path = Path(f"final_dataset/{split}/manifest.csv")
    df = pd.read_csv(manifest_path)

    results = []

    print("=" * 60)
    print(f"PI + MULTI-SCALE ZNCC — {split.upper()}")
    print("=" * 60)

    total_start = time.perf_counter()

    for idx, row in df.iterrows():

        reference_path = row["reference_path"]
        search_path = row["search_path"]

        # Handle paths stored relative to project root
        if not Path(reference_path).exists():
            reference_path = str(Path(".") / reference_path)

        if not Path(search_path).exists():
            search_path = str(Path(".") / search_path)

        reference = cv2.imread(
            reference_path,
            cv2.IMREAD_GRAYSCALE,
        )

        search = cv2.imread(
            search_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if reference is None:
            raise FileNotFoundError(
                f"Could not load reference: {reference_path}"
            )

        if search is None:
            raise FileNotFoundError(
                f"Could not load search: {search_path}"
            )

        start = time.perf_counter()

        match = adaptive_match(
            reference,
            search,
        )

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        pred_x = match["x"]
        pred_y = match["y"]

        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        error = float(
            np.hypot(
                pred_x - gt_x,
                pred_y - gt_y,
            )
        )

        results.append({
            "id": int(row["id"]),
            "architecture": row["architecture"],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "localization_error_px": error,
            "zncc_score": match["score"],
            "scale": match["scale"],
            "iterations": match["iterations"],
            "runtime_ms": runtime_ms,
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)}")

    total_runtime = time.perf_counter() - total_start

    results_df = pd.DataFrame(results)

    errors = results_df["localization_error_px"]

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Samples                  : {len(results_df)}")
    print(f"Mean localization error  : {errors.mean():.3f} px")
    print(f"Median localization error: {errors.median():.3f} px")
    print(f"Maximum localization error: {errors.max():.3f} px")

    print(
        f"< 1 px localization rate : "
        f"{(errors < 1).mean() * 100:.2f}%"
    )

    print(
        f"< 3 px localization rate : "
        f"{(errors < 3).mean() * 100:.2f}%"
    )

    print(
        f"< 5 px localization rate : "
        f"{(errors < 5).mean() * 100:.2f}%"
    )

    print(
        f"> 5 px failure rate       : "
        f"{(errors > 5).mean() * 100:.2f}%"
    )

    print(
        f"> 50 px failure rate      : "
        f"{(errors > 50).mean() * 100:.2f}%"
    )

    print(
        f"> 100 px failure rate     : "
        f"{(errors > 100).mean() * 100:.2f}%"
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
        f"{total_runtime:.3f} s"
    )

    print()
    print("Architecture:")
    print(
        results_df.groupby("architecture")[
            "localization_error_px"
        ].agg(
            ["count", "mean", "median"]
        )
    )

    print()
    print("Worst 20 cases:")

    print(
        results_df.sort_values(
            "localization_error_px",
            ascending=False,
        ).head(20).to_string(index=False)
    )

    output_path = (
        f"pi_validation_results.csv"
        if split == "validation"
        else f"pi_{split}_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":

    split = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "validation"
    )

    evaluate(split)