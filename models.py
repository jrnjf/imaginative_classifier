import torch
import torch.nn as nn
import torchvision

class ConditionalGenerator(nn.Module):
    """
    Takes latent noise z and a class label y, embeds the class label,
    and generates a 32x32 synthetic image conditioned on that class.
    """
    def __init__(self, latent_dim=100, num_classes=100, embed_dim=50):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_dim = latent_dim + embed_dim
        
        self.net = nn.Sequential(
            # [B, in_dim, 1, 1] -> [B, 512, 4, 4]
            nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # [B, 512, 4, 4] -> [B, 256, 8, 8]
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # [B, 256, 8, 8] -> [B, 128, 16, 16]
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # [B, 128, 16, 16] -> [B, 3, 32, 32]
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh() # Scales output to [-1, 1]
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