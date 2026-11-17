import torch
import torch.nn as nn
import torchvision

class ConditionalGenerator(nn.Module):
    """
    Symmetric Double-Cone (Diamond) Conditional Generator (~100.3M parameters).
    Channel progression: 384 -> 1280 -> 2048 (peak) -> 1280 -> 512 -> 3
    Spatial progression: 1x1 -> 4x4  -> 8x8         -> 16x16 -> 32x32 -> 32x32
    """
    def __init__(self, latent_dim=128, num_classes=100, embed_dim=256):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_dim = latent_dim + embed_dim  # 384 channels
        
        self.net = nn.Sequential(
            # --- Cone 1: Expansion ---
            # [B, 384, 1, 1] -> [B, 1280, 4, 4] (7.86M params)
            nn.ConvTranspose2d(in_dim, 1280, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(True),
            
            # Peak Expansion: [B, 1280, 4, 4] -> [B, 2048, 8, 8] (41.94M params)
            nn.ConvTranspose2d(1280, 2048, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            
            # --- Cone 2: Contraction ---
            # [B, 2048, 8, 8] -> [B, 1280, 16, 16] (41.94M params)
            nn.ConvTranspose2d(2048, 1280, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(True),
            
            # [B, 1280, 16, 16] -> [B, 512, 32, 32] (10.49M params)
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Final RGB Projection: [B, 512, 32, 32] -> [B, 3, 32, 32] (13.8K params)
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, c], dim=1)
        return self.net(x)

class DualHeadResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.class_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.truth_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 1) # Outputs raw logit [-inf, inf]
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.class_head(features)
        truth_logit = self.truth_head(features)
        return class_logits, truth_logit