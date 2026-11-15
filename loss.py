import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    """
    Dynamically balances classification loss and Hinge-based truth/realness loss
    using homoscedastic uncertainty log-variances with stability clamping.
    """
    def __init__(self):
        super().__init__()
        self.s1 = nn.Parameter(torch.zeros(1))
        self.s2 = nn.Parameter(torch.zeros(1))
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, class_logits_real, real_labels, truth_logit_real,
                class_logits_fake, truth_logit_fake):
        
        # Clamp log-variances to prevent collapse/explosion (-3.0 to 3.0)
        s1 = torch.clamp(self.s1, min=-3.0, max=3.0)
        s2 = torch.clamp(self.s2, min=-3.0, max=3.0)
        
        # 1. Classification Loss (Real Data)
        loss_class = self.ce_loss(class_logits_real, real_labels)
        
        # 2. Hinge Loss for Truth Head (Discriminator view)
        # Real data logits should be >= +1, Fake data logits should be <= -1
        loss_truth_real = torch.mean(torch.relu(1.0 - truth_logit_real))
        loss_truth_fake = torch.mean(torch.relu(1.0 + truth_logit_fake))
        loss_truth = loss_truth_real + loss_truth_fake
        
        # 3. Precision weighting via clamped log-variance
        precision_class = torch.exp(-s1)
        precision_truth = torch.exp(-s2)
        
        weighted_loss_class = precision_class * loss_class + 0.5 * s1
        weighted_loss_truth = precision_truth * loss_truth + 0.5 * s2
        
        total_loss = weighted_loss_class + weighted_loss_truth
        
        return total_loss, loss_class.item(), loss_truth.item(), s1.item(), s2.item()