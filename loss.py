import torch
import torch.nn as nn
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()
class UncertaintyWeightedLoss(nn.Module):
    """
    Extends uncertainty weighting across both Classifier and Generator 
    using 4 independent log-variances (s1, s2, s3, s4) or an optional Meta-DNN.
    """
    def __init__(self, use_dnn_meta=False):
        super().__init__()
        self.use_dnn_meta = use_dnn_meta
        
        if self.use_dnn_meta:
            # Optional tiny DNN to dynamically output s1, s2, s3, s4 based on training state
            # Using ReLU(inplace=False) to prevent PyTorch inplace modification version errors
            self.meta_net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(inplace=False),
                nn.Linear(16, 16),
                nn.ReLU(inplace=False),
                nn.Linear(16, 4) # Outputs s1, s2, s3, s4
            )
        else:
            # Independent learnable log-variances
            self.s1 = nn.Parameter(torch.zeros(1))
            self.s2 = nn.Parameter(torch.zeros(1))
            self.s3 = nn.Parameter(torch.zeros(1))
            self.s4 = nn.Parameter(torch.zeros(1))
            
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_G = FocalBCEWithLogitsLoss(gamma=2.0, alpha=0.25)
        self.focal_c = FocalBCEWithLogitsLoss(gamma=2.0, alpha=0.75)
    def forward(self, 
                class_logits_real, real_labels, truth_logit_real,
                class_logits_fake, truth_logit_fake,
                class_logits_gen, fake_labels, truth_logit_gen,
                epoch_progress=0.0):
        
        # 1. Compute Base Component Losses
        l_class_real = self.ce_loss(class_logits_real, real_labels)
        l_truth_real = self.focal_c(truth_logit_real, torch.ones_like(truth_logit_real))
        l_truth_fake = self.focal_c(truth_logit_fake, torch.zeros_like(truth_logit_fake))
        l_truth = l_truth_real + l_truth_fake
        
        l_gen_class = self.ce_loss(class_logits_gen, fake_labels)
        l_gen_truth = self.focal_G(truth_logit_gen, torch.ones_like(truth_logit_gen))
        
        # 2. Get Uncertainty Weights (s1, s2, s3, s4)
        if self.use_dnn_meta:
            # Stack and add a batch dimension of size 1: shape becomes [1, 4]
            loss_vector = torch.stack([
                            l_class_real.detach(), 
                            l_truth.detach(), 
                            l_gen_class.detach(), 
                            l_gen_truth.detach()
                        ]).unsqueeze(0).detach()  # Shape becomes [1, 4]
                        
            s_vals = self.meta_net(loss_vector) # Shape [1, 4]
            s1, s2, s3, s4 = s_vals[0, 0], s_vals[0, 1], s_vals[0, 2], s_vals[0, 3]
        else:
            s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4

        # Clamp to prevent numerical explosion/collapse [-3.0, 3.0]
        """
        s1 = torch.clamp(s1, min=-3.0, max=3.0)
        s2 = torch.clamp(s2, min=-3.0, max=3.0)
        s3 = torch.clamp(s3, min=-3.0, max=3.0)
        s4 = torch.clamp(s4, min=-3.0, max=3.0)
        """
        # 3. Uncertainty-Weighted Multi-Task Formulation
        weighted_loss_C = (torch.exp(-s1) * l_class_real + 0.5 * s1) + \
                          (torch.exp(-s2) * l_truth + 0.5 * s2)
                          
        weighted_loss_G = (torch.exp(-s3) * l_gen_class + 0.5 * s3) + \
                          (torch.exp(-s4) * l_gen_truth + 0.5 * s4)
                          
        return weighted_loss_C, weighted_loss_G, {
            'l_class': l_class_real.item(),
            'l_truth': l_truth.item(),
            'l_gen_class': l_gen_class.item(),
            'l_gen_truth': l_gen_truth.item(),
            's1': s1.item(), 's2': s2.item(), 's3': s3.item(), 's4': s4.item()
        }