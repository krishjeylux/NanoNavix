import math
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from baseline_solution.topk_matcher import _extract_peaks

class DeepFeatureExtractor:
    def __init__(self, layer_name="layer3"):
        # Force CPU as per instructions to avoid Apple Silicon MPS issues with specific operations
        # Actually, MPS is allowed: "if torch.backends.mps.is_available(): device = torch.device('mps')"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze the backbone completely
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.layer_name = layer_name
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_feature_map(self, img_np):
        """
        img_np: grayscale image (H, W), values in [0, 255] or [0, 1].
        Returns a numpy array of shape (C, H_f, W_f)
        """
        if img_np.max() > 1.0:
            img_np = img_np.astype(np.float32) / 255.0
        else:
            img_np = img_np.astype(np.float32)
            
        # Grayscale -> 3 identical channels
        img_3c = np.stack([img_np, img_np, img_np], axis=0) # (3, H, W)
        
        tensor = torch.from_numpy(img_3c).to(self.device)
        tensor = self.transform(tensor).unsqueeze(0) # (1, 3, H, W)
        
        with torch.no_grad():
            x = self.model.conv1(tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            if self.layer_name == "layer1":
                return x.squeeze(0).cpu().numpy()
                
            x = self.model.layer2(x)
            if self.layer_name == "layer2":
                return x.squeeze(0).cpu().numpy()
                
            x = self.model.layer3(x)
            if self.layer_name == "layer3":
                return x.squeeze(0).cpu().numpy()
                
            x = self.model.layer4(x)
            if self.layer_name == "layer4":
                return x.squeeze(0).cpu().numpy()
                
        raise ValueError(f"Invalid layer_name: {self.layer_name}")


def generate_deep_candidates(reference: np.ndarray, search: np.ndarray, extractor: DeepFeatureExtractor,
                             scales=(9.0, 9.5, 10.0, 10.5, 11.0), top_n_per_scale=20, 
                             nms_radius=10, guard_radius=10) -> list:
    
    # Extract search features once
    search_feat = extractor.extract_feature_map(search) # (C, H_f, W_f)
    C, h_f, w_f = search_feat.shape
    
    # Infer spatial downsampling factor (stride)
    stride_y = search.shape[0] / h_f
    stride_x = search.shape[1] / w_f
    
    all_peaks = []
    
    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
            
        template_img = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        template_feat = extractor.extract_feature_map(template_img) # (C, h_t, w_t)
        _, h_t, w_t = template_feat.shape
        
        if h_t > h_f or w_t > w_f:
            continue
            
        # We perform channel-wise normalized cross-correlation
        # Aggregation method: Sum the CCOEFF_NORMED responses across all channels, then divide by C.
        corr_map_sum = None
        
        for c in range(C):
            s_c = search_feat[c]
            t_c = template_feat[c]
            
            # cv2.matchTemplate with CCOEFF_NORMED computes the Pearson correlation coefficient.
            res = cv2.matchTemplate(s_c, t_c, cv2.TM_CCOEFF_NORMED)
            
            if corr_map_sum is None:
                corr_map_sum = res
            else:
                corr_map_sum += res
                
        corr_map_avg = corr_map_sum / C
        
        # Handle nan in correlation map (happens if a channel has zero variance)
        corr_map_avg = np.nan_to_num(corr_map_avg, nan=-1.0)
        
        # Feature-space dimensions
        tw_f = w_t
        th_f = h_t
        
        # Map guard radius to feature space
        feature_guard_radius = max(1, int(round(guard_radius / ((stride_x + stride_y) / 2.0))))
        
        peaks_f = _extract_peaks(corr_map_avg, scale, tw_f, th_f, top_n=top_n_per_scale, suppression_radius=feature_guard_radius)
        
        # Map peaks back to pixel space
        for p in peaks_f:
            # p["x"] and p["y"] are the center of the template in feature space
            p["x"] = p["x"] * stride_x
            p["y"] = p["y"] * stride_y
            p["template_w"] = tw
            p["template_h"] = th
            p["source"] = f"deep_{extractor.layer_name}"
            
        all_peaks.extend(peaks_f)
        
    all_peaks.sort(key=lambda p: p["score"], reverse=True)
    
    # Global spatial NMS
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
