import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from baseline_solution.zncc import zncc_match


def evaluate_split(split):
    manifest_path = Path(f"final_dataset/{split}/manifest.csv")
    base_dir = Path(f"final_dataset/{split}")

    df = pd.read_csv(manifest_path)

    errors = []
    scores = []
    runtimes = []

    print("=" * 60)
    print(f"ZNCC BASELINE — {split.upper()}")
    print("=" * 60)

    for i, row in df.iterrows():

        reference_path = Path(row["reference_path"])
        search_path = Path(row["search_path"])

        # Handle paths stored either as relative-to-dataset or full dataset paths
        if not reference_path.is_absolute():
            if str(reference_path).startswith(str(base_dir)):
                ref_path = reference_path
            else:
                ref_path = base_dir / reference_path

        if not search_path.is_absolute():
            if str(search_path).startswith(str(base_dir)):
                search_path = search_path
            else:
                search_path = base_dir / search_path

        reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)

        if reference is None:
            raise FileNotFoundError(f"Reference not found: {ref_path}")

        if search is None:
            raise FileNotFoundError(f"Search not found: {search_path}")

        start = cv2.getTickCount()

        match = zncc_match(reference, search)

        end = cv2.getTickCount()

        runtime_ms = (
            (end - start) / cv2.getTickFrequency()
        ) * 1000.0

        error = np.hypot(
            match["x"] - row["gt_x"],
            match["y"] - row["gt_y"]
        )

        errors.append(error)
        scores.append(match["score"])
        runtimes.append(runtime_ms)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    errors = np.asarray(errors)
    scores = np.asarray(scores)
    runtimes = np.asarray(runtimes)

    print()
    print(f"Samples                  : {len(df)}")
    print(f"Mean localization error  : {errors.mean():.3f} px")
    print(f"Median localization error: {np.median(errors):.3f} px")
    print(f"Maximum localization error: {errors.max():.3f} px")

    print(f"< 1 px localization rate : {(errors < 1).mean() * 100:.2f}%")
    print(f"< 3 px localization rate : {(errors < 3).mean() * 100:.2f}%")
    print(f"< 5 px localization rate : {(errors < 5).mean() * 100:.2f}%")

    print(f"Mean ZNCC score          : {scores.mean():.4f}")
    print(f"Mean runtime             : {runtimes.mean():.3f} ms")
    print(f"Total runtime            : {runtimes.sum() / 1000:.3f} s")

    print()
    print("Architecture:")
    print(
        df.assign(error=errors)
        .groupby("architecture")["error"]
        .agg(["count", "mean", "median"])
    )

    return errors, scores


if __name__ == "__main__":

    split = sys.argv[1] if len(sys.argv) > 1 else "validation"

    evaluate_split(split)