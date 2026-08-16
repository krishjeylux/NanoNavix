import math
import numpy as np
import cv2

from baseline_solution.psr import calculate_psr

def _extract_peaks(correlation_map: np.ndarray, scale: float, tw: int, th: int, top_n: int = 20, suppression_radius: int = 10) -> list:
    """Extracts top_n distinct peaks from a single correlation map using spatial suppression."""
    map_copy = correlation_map.copy()
    peaks = []
    h, w = map_copy.shape
    
    for _ in range(top_n):
        _, max_val, _, max_loc = cv2.minMaxLoc(map_copy)
        if max_val == -np.inf:
            break
            
        x_map, y_map = max_loc
        # Convert to physical search image coordinates
        x_c = x_map + tw / 2.0
        y_c = y_map + th / 2.0
        
        peaks.append({
            "x": x_c,
            "y": y_c,
            "score": float(max_val),
            "scale": scale,
            "template_w": tw,
            "template_h": th,
            "map_x": x_map,
            "map_y": y_map
        })
        
        # Suppress local neighborhood
        y_min = max(0, y_map - suppression_radius)
        y_max = min(h, y_map + suppression_radius + 1)
        x_min = max(0, x_map - suppression_radius)
        x_max = min(w, x_map + suppression_radius + 1)
        map_copy[y_min:y_max, x_min:x_max] = -np.inf
        
    return peaks

def topk_zncc_match(reference: np.ndarray, search: np.ndarray, scales=(9.0, 9.5, 10.0, 10.5, 11.0), top_k=10, nms_radius=25, guard_radius=10, top_n_per_scale=20, return_stats=False):
    """
    Generates Top-K globally distinct candidates across all scales.
    Returns a list of candidate dictionaries.
    """
    all_peaks = []
    correlation_maps = {}
    
    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
            
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        corr_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        
        correlation_maps[scale] = {
            "map": corr_map,
            "tw": tw,
            "th": th
        }
        
        # Internal peak extraction
        peaks = _extract_peaks(corr_map, scale, tw, th, top_n=top_n_per_scale, suppression_radius=guard_radius)
        all_peaks.extend(peaks)
        
    # Sort all peaks across all scales by ZNCC score descending
    all_peaks.sort(key=lambda p: p["score"], reverse=True)
    
    # Global NMS
    candidates = []
    for peak in all_peaks:
        if len(candidates) >= top_k:
            break
            
        # Check distance against already accepted candidates
        is_distinct = True
        for cand in candidates:
            dist = math.hypot(peak["x"] - cand["x"], peak["y"] - cand["y"])
            if dist < nms_radius:
                is_distinct = False
                break
                
        if is_distinct:
            # We accepted this candidate. Compute its advanced features.
            
            # 1. PSR
            corr_map_info = correlation_maps[peak["scale"]]
            psr_stats = calculate_psr(corr_map_info["map"], (peak["map_x"], peak["map_y"]), guard_radius=guard_radius)
            peak["psr"] = psr_stats["psr"]
            
            # 2. Cross-scale consistency
            cross_scores = []
            for s, s_info in correlation_maps.items():
                tw_s, th_s = s_info["tw"], s_info["th"]
                s_map = s_info["map"]
                h, w = s_map.shape
                
                # Map physical coordinates back to correlation map coordinates for scale s
                map_x = int(round(peak["x"] - tw_s / 2.0))
                map_y = int(round(peak["y"] - th_s / 2.0))
                
                map_x = max(0, min(w - 1, map_x))
                map_y = max(0, min(h - 1, map_y))
                
                cross_scores.append(s_map[map_y, map_x])
                
            peak["cross_scale_mean"] = float(np.mean(cross_scores))
            peak["cross_scale_std"] = float(np.std(cross_scores))
            peak["cross_scale_consistency"] = peak["cross_scale_mean"] - peak["cross_scale_std"]
            
            # For peak separation we would need the second highest peak *in that physical region*.
            # For simplicity, we just use PSR instead.
            peak["second_peak_score"] = 0.0 # Placeholder
            peak["peak_separation"] = 0.0 # Placeholder
            
            candidates.append(peak)
            
    if return_stats:
        stats = {
            "pre_nms_candidates": len(all_peaks),
            "post_nms_candidates": len(candidates)
        }
        return candidates, stats
    return candidates

def arbitrate_candidates(candidates: list, mode="full", weights=None) -> dict:
    """
    Arbitrates among Top-K candidates using normalized features.
    
    Modes:
      - zncc_only
      - zncc_psr
      - zncc_psr_scale
      - full
      
    Returns the best candidate according to the arbitration scoring.
    """
    if not candidates:
        return None
        
    if len(candidates) == 1:
        return candidates[0]
        
    if weights is None:
        weights = {
            "zncc": 1.0,
            "psr": 1.0,
            "consistency": 1.0
        }
        
    # Extract features
    znccs = np.array([c["score"] for c in candidates])
    psrs = np.array([c["psr"] for c in candidates])
    cons = np.array([c["cross_scale_consistency"] for c in candidates])
    
    eps = 1e-6
    # Normalize (Z-score normalization)
    zncc_norm = (znccs - np.mean(znccs)) / (np.std(znccs) + eps)
    psr_norm = (psrs - np.mean(psrs)) / (np.std(psrs) + eps)
    cons_norm = (cons - np.mean(cons)) / (np.std(cons) + eps)
    
    final_scores = np.zeros(len(candidates))
    
    if mode == "zncc_only":
        final_scores = znccs # Just use raw score for zncc_only to perfectly match argmax behavior
    elif mode == "zncc_psr":
        final_scores = weights["zncc"] * zncc_norm + weights["psr"] * psr_norm
    elif mode == "zncc_psr_scale" or mode == "full":
        final_scores = weights["zncc"] * zncc_norm + weights["psr"] * psr_norm + weights["consistency"] * cons_norm
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    best_idx = int(np.argmax(final_scores))
    best_candidate = candidates[best_idx].copy()
    best_candidate["arbitration_score"] = float(final_scores[best_idx])
    
    return best_candidate
