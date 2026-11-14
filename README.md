# Imaginative Classifier (CIFAR-100 Pilot)

A hybrid deep learning architecture combining:
1. **Dual-Head ResNet-18 Backbone:** Separates classification ($K$ classes) from real/imaginary truth estimation ($i \in [0, 1]$).
2. **Class-Conditional Generator:** Generates $32 \times 32$ synthetic images conditioned on target classes to act as an active regularizer.
3. **Homoscedastic Uncertainty Weighting:** Dynamically balances the classification and adversarial losses during backpropagation without manual hyperparameter tuning.
