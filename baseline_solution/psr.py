import cv2
import numpy as np

def calculate_psr(correlation_map: np.ndarray, peak_xy: tuple[int, int], guard_radius: int = 5) -> dict:
    """
    Calculates Peak-to-Sidelobe Ratio (PSR) from a correlation map.
    
    Args:
        correlation_map: 2D numpy array of the ZNCC correlation surface.
        peak_xy: (x, y) coordinates of the primary peak in the correlation_map.
        guard_radius: radius of the exclusion region around the peak.
        
    Returns:
        dict containing PSR statistics.
    """
    x, y = int(peak_xy[0]), int(peak_xy[1])
    peak_score = float(correlation_map[y, x])
    
    # Create a boolean mask for the sidelobe region
    mask = np.ones(correlation_map.shape, dtype=bool)
    
    # Exclude the guard region around the peak
    h, w = correlation_map.shape
    y_min = max(0, y - guard_radius)
    y_max = min(h, y + guard_radius + 1)
    x_min = max(0, x - guard_radius)
    x_max = min(w, x + guard_radius + 1)
    
    mask[y_min:y_max, x_min:x_max] = False
    
    sidelobe_pixels = correlation_map[mask]
    
    if len(sidelobe_pixels) == 0:
        return {
            "peak_score": peak_score,
            "sidelobe_mean": 0.0,
            "sidelobe_std": 1e-6,
            "psr": 0.0
        }
        
    sidelobe_mean = float(np.mean(sidelobe_pixels))
    sidelobe_std = float(np.std(sidelobe_pixels))
    
    # Avoid division by zero
    epsilon = 1e-6
    std_safe = max(sidelobe_std, epsilon)
    
    psr = (peak_score - sidelobe_mean) / std_safe
    
    return {
        "peak_score": peak_score,
        "sidelobe_mean": sidelobe_mean,
        "sidelobe_std": sidelobe_std,
        "psr": psr
    }

def extract_top_k_peaks(correlation_map: np.ndarray, k: int = 2, suppression_radius: int = 10) -> list:
    """
    Extracts the top-K distinct peaks from the correlation map using a suppression mask.
    """
    map_copy = correlation_map.copy()
    peaks = []
    
    h, w = map_copy.shape
    
    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(map_copy)
        if max_val == -np.inf:
            break
            
        x, y = max_loc
        peaks.append({"loc": max_loc, "score": float(max_val)})
        
        # Suppress neighborhood
        y_min = max(0, y - suppression_radius)
        y_max = min(h, y + suppression_radius + 1)
        x_min = max(0, x - suppression_radius)
        x_max = min(w, x - suppression_radius + 1) # BUG: should be x + suppression_radius + 1
        x_max = min(w, x + suppression_radius + 1)
        
        map_copy[y_min:y_max, x_min:x_max] = -np.inf
        
    return peaks

def zncc_match_with_psr(reference: np.ndarray, search: np.ndarray, scales=(9.0, 9.5, 10.0, 10.5, 11.0), guard_radius=10) -> dict:
    """
    Multi-scale ZNCC template match. Calculates PSR and extracts distinct peaks.
    """
    best = None
    all_scale_results = []
    
    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
            
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        correlation_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        
        top_peaks = extract_top_k_peaks(correlation_map, k=2, suppression_radius=guard_radius)
        if not top_peaks:
            continue
            
        primary_peak = top_peaks[0]
        psr_stats = calculate_psr(correlation_map, primary_peak["loc"], guard_radius=guard_radius)
        
        second_peak_score = top_peaks[1]["score"] if len(top_peaks) > 1 else None
        
        scale_result = {
            "x": primary_peak["loc"][0] + tw / 2.0,
            "y": primary_peak["loc"][1] + th / 2.0,
            "score": primary_peak["score"],
            "scale": scale,
            "psr": psr_stats["psr"],
            "psr_stats": psr_stats,
            "second_peak_score": second_peak_score,
            "peak_diff": (primary_peak["score"] - second_peak_score) if second_peak_score is not None else None,
            "template_w": tw,
            "template_h": th,
            "max_loc": primary_peak["loc"],
            "correlation_map": correlation_map
        }
        all_scale_results.append(scale_result)
        
        if best is None or scale_result["score"] > best["score"]:
            best = scale_result
            
    return best, all_scale_results
