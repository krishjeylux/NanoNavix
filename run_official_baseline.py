import csv
import os
import sys
import time
import numpy as np
import pandas as pd

from baseline_solution.zncc import zncc_match


MANIFEST = "final_dataset/test/manifest.csv"
OUTPUT = "official_baseline_test_results.csv"


def main():
    df = pd.read_csv(MANIFEST)

    results = []

    print("=" * 70)
    print("ORGANIZER BASELINE — TEST SET")
    print("=" * 70)

    start_total = time.perf_counter()

    for i, row in df.iterrows():

        reference_path = row["reference_path"]
        search_path = row["search_path"]

        # Handle paths stored relative to the dataset directory.
        if not os.path.exists(reference_path):
            reference_path = os.path.join("final_dataset", "test",
                                          reference_path)

        if not os.path.exists(search_path):
            search_path = os.path.join("final_dataset", "test",
                                       search_path)

        reference = __import__("cv2").imread(
            reference_path,
            __import__("cv2").IMREAD_GRAYSCALE
        )

        search = __import__("cv2").imread(
            search_path,
            __import__("cv2").IMREAD_GRAYSCALE
        )

        if reference is None or search is None:
            raise RuntimeError(
                f"Could not read:\n"
                f"Reference: {reference_path}\n"
                f"Search: {search_path}"
            )

        t0 = time.perf_counter()

        match = zncc_match(reference, search)

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        pred_x = match["x"]
        pred_y = match["y"]

        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        error = float(
            np.hypot(
                pred_x - gt_x,
                pred_y - gt_y
            )
        )

        results.append({
            "id": row["id"],
            "architecture": row["architecture"],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "localization_error_px": error,
            "zncc_score": match["score"],
            "scale": match["scale"],
            "template_w": match["template_w"],
            "template_h": match["template_h"],
            "runtime_ms": runtime_ms,
        })

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    total_runtime = time.perf_counter() - start_total

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)

    print()
    print("=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"Samples                  : {len(out)}")
    print(
        f"Mean localization error  : "
        f"{out['localization_error_px'].mean():.3f} px"
    )
    print(
        f"Median localization error: "
        f"{out['localization_error_px'].median():.3f} px"
    )
    print(
        f"Maximum localization error: "
        f"{out['localization_error_px'].max():.3f} px"
    )

    print(
        f"< 1 px localization rate  : "
        f"{(out['localization_error_px'] < 1).mean() * 100:.2f}%"
    )

    print(
        f"> 3 px failure rate        : "
        f"{(out['localization_error_px'] > 3).mean() * 100:.2f}%"
    )

    print(
        f"> 5 px failure rate        : "
        f"{(out['localization_error_px'] > 5).mean() * 100:.2f}%"
    )

    print(
        f"Mean ZNCC score            : "
        f"{out['zncc_score'].mean():.4f}"
    )

    print(
        f"Mean runtime               : "
        f"{out['runtime_ms'].mean():.3f} ms"
    )

    print(
        f"Total runtime              : "
        f"{total_runtime:.3f} s"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()