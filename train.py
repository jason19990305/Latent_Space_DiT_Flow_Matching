import os
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL

from dataset import get_celeba_dataloader
from dit import DiT
from flow_matching import FlowMatcher


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for p_ema, p_model in zip(self.shadow.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p_model.data, alpha=1.0 - self.decay)


def train(num_epochs=3000, lr=1e-3, num_samples=10, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "dit_celeba10.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Load & encode training images to VAE latents
    dataloader = get_celeba_dataloader(num_samples=num_samples)
    images = next(iter(dataloader)).to(device)
    labels = torch.arange(num_samples, device=device)
    print(f"[Train] Loaded {images.shape[0]} CelebA images (labels 0..{num_samples-1}).")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode() * 0.18215
    print(f"[Train] Latents shape: {latents.shape}")

    # Initialize DiT model, EMA, FlowMatcher, and Optimizer
    model = DiT(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=384, depth=8, num_heads=12, num_classes=num_samples
    ).to(device)

    ema = EMA(model, decay=0.999)
    flow_matcher = FlowMatcher(num_train_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Training loop
    print(f"[Train] Training DiT Flow Matching for {num_epochs} epochs...")
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = flow_matcher.compute_loss(model, latents, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if epoch % 300 == 0 or epoch == 1:
            print(f"Epoch [{epoch:04d}/{num_epochs}] | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow.state_dict(),
        "input_size": 32, "patch_size": 2, "in_channels": 4,
        "hidden_size": 384, "depth": 8, "num_heads": 12, "num_classes": num_samples
    }, save_path)
    print(f"[Train] Saved checkpoint with EMA to '{save_path}'.")


if __name__ == "__main__":
    train()

