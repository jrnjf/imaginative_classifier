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
    latent_dim = 100
    total_epochs = 200
    period_epochs = 1.0

    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=100).to(device)
    classifier = DualHeadResNet18(num_classes=100).to(device)
    criterion_uncertainty = UncertaintyWeightedLoss().to(device)

    # Momentum-free optimizers (SGD with momentum=0.0)
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.85, 0.999), weight_decay=1e-2)
    optimizer_C = optim.AdamW([
        {'params': classifier.parameters()},
        {'params': criterion_uncertainty.parameters(), 'lr': 0.0002}
    ], lr=0.0002, betas=(0.85, 0.999), weight_decay=1e-2)

    steps_per_epoch = len(trainloader)
    period_steps = period_epochs * steps_per_epoch

    # ---------------------------------------------------------
    # 3. Schedulers with 90-Degree Phase Shift
    # ---------------------------------------------------------


    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    # ---------------------------------------------------------
    # 4. Training & Validation Loop
    # ---------------------------------------------------------
    print("\nStarting Training with Detailed Step Logging & Validation...")
    
    for epoch in range(total_epochs):
        generator.train()
        classifier.train()

        for i, (real_imgs, real_labels) in enumerate(trainloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # --- Step A: Train Dual-Head Classifier ---
# --- Step A: Train Dual-Head Classifier ---
            optimizer_C.zero_grad()
            class_logits_real, truth_logit_real = classifier(real_imgs)

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, 100, (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels).detach()

            class_logits_fake, truth_logit_fake = classifier(fake_imgs)

            loss_C, l_class, l_truth, s1, s2 = criterion_uncertainty(
                class_logits_real, real_labels, truth_logit_real,
                class_logits_fake, truth_logit_fake
            )

            loss_C.backward()
            optimizer_C.step()

            # --- Step B: Train Conditional Generator (Decoupled Reward) ---
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z, fake_labels)
            class_logits_gen, truth_logit_gen = classifier(gen_imgs)

            # 1. Classification Reward: Rewarded for hitting the correct target class 
            # (even if the classifier's truth head catches it as fake)
            loss_G_class = ce_criterion(class_logits_gen, fake_labels)

            # 2. Hinge Deception Loss: Generator wants truth logit to look real (>= +1)
            loss_G_truth = -torch.mean(truth_logit_gen)

            # Total Generator Loss combining category learning and deception
            loss_G = loss_G_class + 0.5 * loss_G_truth
            loss_G.backward()
            optimizer_G.step()
            # --- Per-Step Detailed Logging ---
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{total_epochs}], Step [{i}/{steps_per_epoch}]: "
                      f"Class Loss: {l_class:.4f} | Truth Loss: {l_truth:.4f} | "
                      f"Gen Loss: {loss_G.item():.4f} | weights [s1: {s1:.2f}, s2: {s2:.2f}]")

        # ---------------------------------------------------------
        # 5. Validation Evaluation on Real Dataset Split
        # ---------------------------------------------------------
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_imgs, val_labels in testloader:
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                val_class_logits, val_truth_prob = classifier(val_imgs)
                
                ce_val = ce_criterion(val_class_logits, val_labels)
                bce_val = bce_criterion(val_truth_prob, torch.ones_like(val_truth_prob))
                batch_val_loss = ce_val + bce_val
                
                val_loss += batch_val_loss.item() * val_imgs.size(0)
                
                _, predicted = torch.max(val_class_logits.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        epoch_val_loss = val_loss / len(testset)
        epoch_val_acc = 100.0 * correct / total

        print(f"============================================================")
        print(f"EPOCH [{epoch+1}/{total_epochs}] COMPLETE | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"============================================================")

    print("\nTraining and validation complete!")

if __name__ == '__main__':
    main()