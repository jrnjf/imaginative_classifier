import torch
import torch.nn as nn

def relative_factor(x1,x2):
    y1 = x1/x2 - x2/x1
    y2 = -y1
    return torch.relu(1+y1), torch.relu(1+y2)

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
        

            
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_G = FocalBCEWithLogitsLoss(gamma=2.0, alpha=0.25)
        self.focal_c = FocalBCEWithLogitsLoss(gamma=2.0, alpha=0.75)
    def forward(self, 
                class_logits_real, real_labels, truth_logit_real,
                class_logits_fake, truth_logit_fake,
                class_logits_gen, fake_labels, truth_logit_gen,
                epoch_progress=0.0):
        
        # 1. Compute Base Component Losses
        truth_logits = torch.cat([truth_logit_real, truth_logit_fake], dim=0)
        truth_targets = torch.cat([torch.ones_like(truth_logit_real), torch.zeros_like(truth_logit_fake)], dim=0)
        l_class_real = self.ce_loss(class_logits_real, real_labels)
        l_truth = self.focal_c(truth_logits, truth_targets)

        
        l_gen_class = self.ce_loss(class_logits_gen, fake_labels)
        l_gen_truth = self.focal_G(truth_logit_gen, torch.ones_like(truth_logit_gen))

        # 2. Get relative Weights (s1, s2, s3, s4)

        fc_1, fc_2 = relative_factor(l_class_real, l_truth)
        fg_1, fg_2 = relative_factor(l_gen_class, l_gen_truth)
        fcg_1_class,fcg_2_class = relative_factor(l_class_real,l_gen_class)
        fcg_1_truth,fcg_2_truth = relative_factor(l_truth,l_gen_truth)


        #3. calculate weighted loss
        weighted_loss_C = torch.exp(-fcg_2_class) * (torch.exp(-fc_2) * l_class_real + torch.exp(-fcg_2_truth)*l_gen_class)   + \
                          torch.exp(-fcg_2_truth) * (torch.exp(-fc_1) * l_truth )
                          
        weighted_loss_G = torch.exp(-fcg_1_class) * (torch.exp(-fg_2) * l_gen_class ) + \
                          torch.exp(-fcg_1_truth) * (torch.exp(-fg_1) * l_gen_truth )
        return weighted_loss_C, weighted_loss_G, {
            'l_class': l_class_real.item(),
            'l_truth': l_truth.item(),
            'l_gen_class': l_gen_class.item(),
            'l_gen_truth': l_gen_truth.item(),
            's1': [fc_1.item(), fc_2.item()],
            's2': [fg_1.item(), fg_2.item()],
            's3': [fcg_1_class.item(), fcg_2_class.item()],
            's4': [fcg_1_truth.item(), fcg_2_truth.item()],
            'classifier_class_factor': [torch.exp(-fcg_2_class) , torch.exp(-fc_2), torch.exp(-fcg_2_truth)] ,
            'classifier_truth_factor': [torch.exp(-fcg_2_truth) , torch.exp(-fc_1)],
            'generator_class_factor': [torch.exp(-fcg_1_class) *  torch.exp(-fg_2)],
            'generator_truth_factor': [torch.exp(-fcg_1_truth),fg_1] 
        }

        