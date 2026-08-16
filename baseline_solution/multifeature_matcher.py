import math
import numpy as np
import cv2

from baseline_solution.topk_matcher import _extract_peaks

def compute_edge_representation(img: np.ndarray) -> np.ndarray:
    """Computes Sobel gradient magnitude."""
    img_f32 = img.astype(np.float32)
    gx = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag

def compute_highpass_representation(img: np.ndarray, kernel_size=9) -> np.ndarray:
    """Computes high-pass representation: img - GaussianBlur(img)."""
    img_f32 = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f32, (kernel_size, kernel_size), 0)
    highpass = img_f32 - blurred
    return highpass

def generate_candidates(reference: np.ndarray, search: np.ndarray, feature_type="intensity", 
                       scales=(9.0, 9.5, 10.0, 10.5, 11.0), top_n_per_scale=20, 
                       nms_radius=10, guard_radius=10) -> list:
    """Generates candidates using a specific feature representation."""
    if feature_type == "intensity":
        ref_feat = reference
        search_feat = search
    elif feature_type == "edge":
        ref_feat = compute_edge_representation(reference)
        search_feat = compute_edge_representation(search)
    elif feature_type == "highpass":
        ref_feat = compute_highpass_representation(reference)
        search_feat = compute_highpass_representation(search)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
        
    all_peaks = []
    
    for scale in scales:
        tw = max(int(round(ref_feat.shape[1] / scale)), 1)
        th = max(int(round(ref_feat.shape[0] / scale)), 1)
        if tw >= search_feat.shape[1] or th >= search_feat.shape[0]:
            continue
            
        template = cv2.resize(ref_feat, (tw, th), interpolation=cv2.INTER_AREA)
        corr_map = cv2.matchTemplate(search_feat, template, cv2.TM_CCOEFF_NORMED)
        
        peaks = _extract_peaks(corr_map, scale, tw, th, top_n=top_n_per_scale, suppression_radius=guard_radius)
        for p in peaks:
            p["source"] = feature_type
        all_peaks.extend(peaks)
        
    all_peaks.sort(key=lambda p: p["score"], reverse=True)
    
    candidates = []
    for peak in all_peaks:
        is_distinct = True
        for cand in candidates:
            dist = math.hypot(peak["x"] - cand["x"], peak["y"] - cand["y"])
            if dist < nms_radius:
                is_distinct = False
                break
                
        if is_distinct:
            candidates.append(peak)
            
    return candidates

def combine_and_deduplicate(pools: list, nms_radius=10) -> list:
    """
    Combines candidate pools from different features and applies spatial NMS.
    Candidates are prioritized by their original ZNCC score.
    """
    all_cands = []
    for pool in pools:
        all_cands.extend(pool)
        
    all_cands.sort(key=lambda p: p["score"], reverse=True)
    
    deduped = []
    for peak in all_cands:
        is_distinct = True
        for cand in deduped:
            dist = math.hypot(peak["x"] - cand["x"], peak["y"] - cand["y"])
            if dist < nms_radius:
                is_distinct = False
                break
                
        if is_distinct:
            deduped.append(peak)
            
    return deduped
