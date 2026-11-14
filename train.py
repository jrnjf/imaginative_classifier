import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import ConditionalGenerator, DualHeadResNet18
from loss import UncertaintyWeightedLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Data Loading (Zero Data Augmentation as requested)
    # ---------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 stats
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

    # ---------------------------------------------------------
    # Model & Loss Initialization
    # ---------------------------------------------------------
    latent_dim = 100
    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=100).to(device)
    classifier = DualHeadResNet18(num_classes=100).to(device)
    criterion_uncertainty = UncertaintyWeightedLoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = optim.Adam([
        {'params': classifier.parameters()},
        {'params': criterion_uncertainty.parameters(), 'lr': 0.001}
    ], lr=0.0002, betas=(0.5, 0.999))

    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    generator.train()
    classifier.train()

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    print("\nStarting Imaginative Classifier Training Pilot...")
    for epoch in range(1): # Adjust number of epochs as needed
        for i, (real_imgs, real_labels) in enumerate(trainloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # -------------------------------------------------
            # Step A: Train Dual-Head Classifier (Discriminator)
            # -------------------------------------------------
            optimizer_C.zero_grad()

            # 1. Real Images Forward Pass
            class_logits_real, truth_prob_real = classifier(real_imgs)
            real_truth_targets = torch.ones_like(truth_prob_real)

            # 2. Generated Images Forward Pass
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, 100, (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels).detach()

            class_logits_fake, truth_prob_fake = classifier(fake_imgs)
            fake_truth_targets = torch.zeros_like(truth_prob_fake)

            # 3. Uncertainty-Weighted Loss Computation
            loss_C, l_class, l_truth, s1, s2 = criterion_uncertainty(
                class_logits_real, real_labels, truth_prob_real, real_truth_targets,
                class_logits_fake, truth_prob_fake, fake_truth_targets
            )

            loss_C.backward()
            optimizer_C.step()

            # -------------------------------------------------
            # Step B: Train Conditional Generator
            # -------------------------------------------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z, fake_labels)
            class_logits_gen, truth_prob_gen = classifier(gen_imgs)

            # Generator Loss: Class reward (CE) + Realness deception reward (BCE targeting 1.0)
            loss_G_class = ce_criterion(class_logits_gen, fake_labels)
            gen_fool_targets = torch.ones_like(truth_prob_gen)
            loss_G_truth = bce_criterion(truth_prob_gen, gen_fool_targets)

            loss_G = loss_G_class + 0.5 * loss_G_truth
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/1], Step [{i}/{len(trainloader)}]: "
                      f"Class Loss: {l_class:.4f} | Truth Loss: {l_truth:.4f} | "
                      f"Gen Loss: {loss_G.item():.4f} | weights [s1: {s1:.2f}, s2: {s2:.2f}]")

    print("\nPilot training iteration completed successfully!")

if __name__ == '__main__':
    main()
