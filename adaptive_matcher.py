#!/usr/bin/env python3

import argparse
import cv2
import numpy as np


class PIController:
    def __init__(
        self,
        kp=0.15,
        ki=0.04,
        target_confidence=0.90,
        min_window=20,
        max_window=200,
    ):
        self.kp = kp
        self.ki = ki
        self.target_confidence = target_confidence
        self.min_window = min_window
        self.max_window = max_window
        self.integral = 0.0

    def update(self, current_window, confidence):
        error = self.target_confidence - confidence

        self.integral += error
        self.integral = np.clip(self.integral, -10.0, 10.0)

        adjustment = (
            self.kp * error
            + self.ki * self.integral
        )

        new_window = current_window * (1.0 + adjustment)

        return float(
            np.clip(
                new_window,
                self.min_window,
                self.max_window,
            )
        )


def global_match(reference, search, scales):
    """
    Initial coarse localization over the complete search image.
    """

    best = None

    for scale in scales:

        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)

        if tw >= search.shape[1] or th >= search.shape[0]:
            continue

        template = cv2.resize(
            reference,
            (tw, th),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, max_loc = cv2.minMaxLoc(result)

        candidate = {
            "x": max_loc[0] + tw / 2.0,
            "y": max_loc[1] + th / 2.0,
            "score": float(score),
            "scale": float(scale),
            "template_w": tw,
            "template_h": th,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def local_match(
    reference,
    search,
    center_x,
    center_y,
    window,
    scales,
):
    """
    Search only inside a local window around the current estimate.
    """

    h, w = search.shape

    x1 = max(int(center_x - window), 0)
    y1 = max(int(center_y - window), 0)
    x2 = min(int(center_x + window), w)
    y2 = min(int(center_y + window), h)

    roi = search[y1:y2, x1:x2]

    best = None

    for scale in scales:

        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)

        if tw >= roi.shape[1] or th >= roi.shape[0]:
            continue

        template = cv2.resize(
            reference,
            (tw, th),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.matchTemplate(
            roi,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, max_loc = cv2.minMaxLoc(result)

        px = x1 + max_loc[0] + tw / 2.0
        py = y1 + max_loc[1] + th / 2.0

        candidate = {
            "x": px,
            "y": py,
            "score": float(score),
            "scale": float(scale),
            "template_w": tw,
            "template_h": th,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def adaptive_localize(
    reference,
    search,
    scales=(9.0, 9.5, 10.0, 10.5, 11.0),
    iterations=6,
    initial_window=100.0,
):

    # ---------------------------------------------------------
    # STEP 1: GLOBAL COARSE LOCALIZATION
    # ---------------------------------------------------------

    current = global_match(
        reference,
        search,
        scales,
    )

    if current is None:
        raise RuntimeError("Could not obtain initial global match")

    x = current["x"]
    y = current["y"]

    controller = PIController()

    window = initial_window

    history = []

    history.append({
        "iteration": 0,
        "x": x,
        "y": y,
        "score": current["score"],
        "scale": current["scale"],
        "window": window,
        "type": "global",
    })

    # ---------------------------------------------------------
    # STEP 2: LOCAL PI-GUIDED REFINEMENT
    # ---------------------------------------------------------

    for iteration in range(1, iterations + 1):

        candidate = local_match(
            reference,
            search,
            x,
            y,
            window,
            scales,
        )

        if candidate is None:
            break

        new_x = candidate["x"]
        new_y = candidate["y"]
        confidence = candidate["score"]

        new_window = controller.update(
            window,
            confidence,
        )

        history.append({
            "iteration": iteration,
            "x": new_x,
            "y": new_y,
            "score": confidence,
            "scale": candidate["scale"],
            "window": new_window,
            "type": "local",
        })

        x = new_x
        y = new_y
        window = new_window

    return {
        "x": x,
        "y": y,
        "score": history[-1]["score"],
        "scale": history[-1]["scale"],
        "iterations": len(history) - 1,
        "history": history,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--search",
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--window",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    reference = cv2.imread(
        args.reference,
        cv2.IMREAD_GRAYSCALE,
    )

    search = cv2.imread(
        args.search,
        cv2.IMREAD_GRAYSCALE,
    )

    if reference is None:
        raise ValueError(
            f"Could not read reference: {args.reference}"
        )

    if search is None:
        raise ValueError(
            f"Could not read search: {args.search}"
        )

    result = adaptive_localize(
        reference,
        search,
        iterations=args.iterations,
        initial_window=args.window,
    )

    print("=" * 60)
    print("PI + LOCAL MULTI-SCALE ZNCC")
    print("=" * 60)

    print(f"Predicted X : {result['x']:.2f}")
    print(f"Predicted Y : {result['y']:.2f}")
    print(f"ZNCC score  : {result['score']:.4f}")
    print(f"Scale       : {result['scale']}")
    print(f"Iterations  : {result['iterations']}")

    print("\nController history:")

    for h in result["history"]:

        print(
            f"Iteration {h['iteration']}: "
            f"x={h['x']:.2f}, "
            f"y={h['y']:.2f}, "
            f"score={h['score']:.4f}, "
            f"window={h['window']:.2f}px, "
            f"type={h['type']}"
        )


if __name__ == "__main__":
    main()
