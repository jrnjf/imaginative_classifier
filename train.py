import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import ConditionalGenerator, DualHeadResNet18
from loss import UncertaintyWeightedLoss
from scheduler import TriangularCosineLR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. Data Loading (Zero Data Augmentation as requested)
    # ---------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 stats
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # ---------------------------------------------------------
    # 2. Model, Loss & Momentum-Free Optimizer Initialization
    # ---------------------------------------------------------
    latent_dim = 128  # Matched to your 100M diamond generator dimension
    total_epochs = 200
    period_epochs = 1.0

    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=100).to(device)
    classifier = DualHeadResNet18(num_classes=100).to(device)
    criterion_uncertainty = UncertaintyWeightedLoss(use_dnn_meta=False).to(device)
    
    # Momentum-free optimizers (SGD / AdamW configurations)
    if criterion_uncertainty.use_dnn_meta:
        optimizer_C = optim.SGD([
            {'params': classifier.parameters()},
            {'params': criterion_uncertainty.meta_net.parameters(), 'lr': 0.0}
        ], lr=1.0, momentum=0.0, weight_decay=1e-3)
        optimizer_G = optim.SGD([
            {'params': generator.parameters()},
            {'params': criterion_uncertainty.meta_net.parameters(), 'lr': 0.0}
        ], lr=1.0, momentum=0.0, weight_decay=1e-4)
    else:
        optimizer_C = optim.SGD(classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-9)
        optimizer_G = optim.SGD(generator.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-9)

    steps_per_epoch = len(trainloader)
    period_steps = period_epochs * steps_per_epoch
    if criterion_uncertainty.use_dnn_meta:
        scheduler_c = TriangularCosineLR(optimizer_C, period_steps=period_steps, total_steps=total_epochs*steps_per_epoch, lr_ratio=1000.0, cos_end_ratio=0.1)
        scheduler_g = TriangularCosineLR(optimizer_G, period_steps=period_steps, total_steps=total_epochs*steps_per_epoch, lr_ratio=1000.0, cos_end_ratio=0.1) #, phase_steps=period_steps/2.0) 
    
    ce_criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # 3. Training & Validation Loop
    # ---------------------------------------------------------
    print("\nStarting Training with Detailed Step Logging & Validation...")
    
    for epoch in range(total_epochs):
        generator.train()
        classifier.train()

        for i, (real_imgs, real_labels) in enumerate(trainloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # --- Forward Passes ---
            class_logits_real, truth_logit_real = classifier(real_imgs)

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, 100, (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels).detach()
            class_logits_fake, truth_logit_fake = classifier(fake_imgs)


            # --- Compute Unified Uncertainty Losses (s1, s2, s3, s4) ---
            loss_C, loss_G, metrics = criterion_uncertainty(
                class_logits_real, real_labels, truth_logit_real,
                class_logits_fake, truth_logit_fake,
                class_logits_fake, fake_labels, truth_logit_fake
            )

            # --- Step A: Train Classifier (Conditional on truth loss >= 1.2) ---
            optimizer_C.zero_grad()
            optimizer_G.zero_grad()

            loss_C.backward(retain_graph=True)  # Retain graph so loss_G can backprop through shared weights/meta_net
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0) # <--- Add this
            # --- Step B: Train Conditional Generator ---
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) # <--- Add this
            optimizer_G.step()
            scheduler_g.step() if criterion_uncertainty.use_dnn_meta else None
            optimizer_C.step()
            scheduler_c.step() if criterion_uncertainty.use_dnn_meta else None

            

            # --- Per-Step Detailed Logging ---
            if i % 100 == 0 :
                print(f"Epoch [{epoch+1}/{total_epochs}], Step [{i}/{steps_per_epoch}]: "
                      f"Class Loss: {metrics['l_class']:.4f} | Truth Loss: {metrics['l_truth']:.4f} | "
                      f"Gen Class: {metrics['l_gen_class']:.4f} | Gen Truth: {metrics['l_gen_truth']:.4f} | "
                      f"weights [fc: {metrics['s1'][0]:.2f},{metrics['s1'][1]:.2f} fg: {metrics['s2'][0]:.2f},{metrics['s2'][1]:.2f}, fcg class: {metrics['s3'][0]:.2f},{metrics['s3'][1]:.2f}, fcg truth: {metrics['s4'][0]:.2f},{metrics['s4'][1]:.2f}] | ")
        # ---------------------------------------------------------
        # Validation Loop (Standard Cross-Entropy & Accuracy)
        # ---------------------------------------------------------
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_imgs, val_labels in testloader:
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                val_class_logits, _ = classifier(val_imgs)
                loss = ce_criterion(val_class_logits, val_labels)
                
                val_loss += loss.item() * val_imgs.size(0)
                _, predicted = torch.max(val_class_logits.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        epoch_val_loss = val_loss / total
        epoch_val_acc = 100.0 * correct / total
        
        print(f"============================================================")
        print(f"EPOCH [{epoch+1}/{total_epochs}] COMPLETE | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"============================================================")

    print("\nTraining and validation complete!")

if __name__ == '__main__':
    main()