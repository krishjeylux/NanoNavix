import os
import cv2
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle

from baseline_solution.topk_matcher import topk_zncc_match

def extract_patch(image, x, y, w, h):
    """Extracts a patch from image centered at (x, y) with size (w, h).
       Pads with zeros if out of bounds."""
    H, W = image.shape
    x_min = int(round(x - w / 2))
    x_max = x_min + w
    y_min = int(round(y - h / 2))
    y_max = y_min + h
    
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - W)
    pad_bottom = max(0, y_max - H)
    
    x_min_valid = max(0, x_min)
    y_min_valid = max(0, y_min)
    x_max_valid = min(W, x_max)
    y_max_valid = min(H, y_max)
    
    patch = image[y_min_valid:y_max_valid, x_min_valid:x_max_valid]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return patch

def load_image_robustly(project_root, split, image_path):
    if os.path.isabs(image_path):
        path = image_path
    elif image_path.startswith("./final_dataset") or image_path.startswith("final_dataset"):
        if image_path.startswith("./"):
            path = os.path.join(project_root, image_path[2:])
        else:
            path = os.path.join(project_root, image_path)
    else:
        path = os.path.join(project_root, "final_dataset", split, image_path)
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

class SiameseCandidateDataset(Dataset):
    def __init__(self, project_root, split, cache_file=None):
        self.project_root = project_root
        self.split = split
        self.manifest_path = os.path.join(project_root, "final_dataset", split, "manifest.csv")
        self.df = pd.read_csv(self.manifest_path)
        self.samples = [] 
        self.stats = None
        
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                self.samples = data["samples"]
                self.stats = data.get("stats", None)
        else:
            print(f"Generating candidate patches for {split}...")
            self._generate_dataset()
            if cache_file:
                with open(cache_file, "wb") as f:
                    pickle.dump({"samples": self.samples, "stats": getattr(self, "stats", None)}, f)
                    
        if split == "train" and self.stats is None:
            self._compute_stats()
            if cache_file:
                with open(cache_file, "wb") as f:
                    pickle.dump({"samples": self.samples, "stats": self.stats}, f)

    def _generate_dataset(self):
        for i, row in self.df.iterrows():
            sample_id = int(row["id"])
            gt_x = float(row["gt_x"])
            gt_y = float(row["gt_y"])
            
            ref = load_image_robustly(self.project_root, self.split, row["reference_path"])
            search = load_image_robustly(self.project_root, self.split, row["search_path"])
            
            cands = topk_zncc_match(ref, search, top_k=50, nms_radius=10)
            
            pos_cand = None
            neg_cands = []
            
            for c in cands:
                err = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
                if err <= 5.0 and pos_cand is None:
                    pos_cand = c
                elif err > 5.0:
                    neg_cands.append(c)
                    
            if pos_cand is None:
                tw = max(int(round(ref.shape[1] / 10.0)), 1)
                th = max(int(round(ref.shape[0] / 10.0)), 1)
                pos_cand = {
                    "x": gt_x, "y": gt_y, "scale": 10.0,
                    "template_w": tw, "template_h": th,
                    "score": 0.5, 
                    "psr": 5.0,
                    "cross_scale_consistency": 0.0
                }
                
            neg_cands.sort(key=lambda c: c["score"], reverse=True)
            neg_cands = neg_cands[:10]
            
            def create_sample_dict(c, label):
                ref_resized = cv2.resize(ref, (64, 64), interpolation=cv2.INTER_AREA)
                search_patch = extract_patch(search, c["x"], c["y"], c["template_w"], c["template_h"])
                search_resized = cv2.resize(search_patch, (64, 64), interpolation=cv2.INTER_AREA)
                
                return {
                    "sample_id": sample_id,
                    "ref_patch": ref_resized,
                    "search_patch": search_resized,
                    "label": label,
                    "features": [c["score"], c.get("psr", 0.0), c.get("cross_scale_consistency", 0.0)]
                }
                
            self.samples.append(create_sample_dict(pos_cand, 1.0))
            for nc in neg_cands:
                self.samples.append(create_sample_dict(nc, 0.0))
                
    def _compute_stats(self):
        features = np.array([s["features"] for s in self.samples])
        self.stats = {
            "mean": features.mean(axis=0).tolist(),
            "std": features.std(axis=0).tolist()
        }
        for i in range(len(self.stats["std"])):
            if self.stats["std"][i] < 1e-6:
                self.stats["std"][i] = 1.0

    def get_stats(self):
        return self.stats
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        ref = s["ref_patch"].astype(np.float32) / 255.0
        search = s["search_patch"].astype(np.float32) / 255.0
        
        ref = torch.from_numpy(ref).unsqueeze(0)
        search = torch.from_numpy(search).unsqueeze(0)
        
        label = torch.tensor([s["label"]], dtype=torch.float32)
        
        feats = np.array(s["features"], dtype=np.float32)
        if self.stats is not None:
            mean = np.array(self.stats["mean"])
            std = np.array(self.stats["std"])
            feats = (feats - mean) / std
            
        feats = torch.from_numpy(feats).float()
        return ref, search, feats, label
