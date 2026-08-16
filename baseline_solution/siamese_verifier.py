import torch
import torch.nn as nn

class SiameseVerifier(nn.Module):
    def __init__(self, mode="cnn_plus_classical"):
        super().__init__()
        self.mode = mode
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_cnn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        if self.mode == "cnn_only":
            self.head = nn.Linear(32, 1)
        elif self.mode == "classical_only":
            self.head = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        elif self.mode == "cnn_plus_classical":
            self.head = nn.Sequential(
                nn.Linear(32 + 3, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, ref, search, classical_feats=None):
        if self.mode != "classical_only":
            emb_ref = self.encoder(ref).view(ref.size(0), -1)
            emb_search = self.encoder(search).view(search.size(0), -1)
            
            diff = torch.abs(emb_ref - emb_search)
            cnn_out = self.fc_cnn(diff)
            
        if self.mode == "cnn_only":
            return self.head(cnn_out)
        elif self.mode == "classical_only":
            return self.head(classical_feats)
        elif self.mode == "cnn_plus_classical":
            combined = torch.cat([cnn_out, classical_feats], dim=1)
            return self.head(combined)
