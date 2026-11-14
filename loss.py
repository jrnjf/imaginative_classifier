import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    """
    Dynamically balances classification loss and truth/realness loss
    using homoscedastic uncertainty log-variances (Kendall et al.).
    """
    def __init__(self):
        super().__init__()
        self.s1 = nn.Parameter(torch.zeros(1)) # Log-variance for classification task
        self.s2 = nn.Parameter(torch.zeros(1)) # Log-variance for truth/realness task
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, class_logits_real, real_labels, truth_prob_real, real_truth_targets,
                class_logits_fake, truth_prob_fake, fake_truth_targets):
        
        # 1. Component losses
        loss_class = self.ce_loss(class_logits_real, real_labels)
        
        loss_truth_real = self.bce_loss(truth_prob_real, real_truth_targets)
        loss_truth_fake = self.bce_loss(truth_prob_fake, fake_truth_targets)
        loss_truth = loss_truth_real + loss_truth_fake
        
        # 2. Precision weighting via log-variance
        precision_class = torch.exp(-self.s1)
        precision_truth = torch.exp(-self.s2)
        
        weighted_loss_class = precision_class * loss_class + 0.5 * self.s1
        weighted_loss_truth = precision_truth * loss_truth + 0.5 * self.s2
        
        total_loss = weighted_loss_class + weighted_loss_truth
        
        return total_loss, loss_class.item(), loss_truth.item(), self.s1.item(), self.s2.item()
