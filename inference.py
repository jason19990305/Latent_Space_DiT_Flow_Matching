import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from diffusers import AutoencoderKL

from dit import DiT


def load_vae(device):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def latent_to_pil(latent, vae):
    with torch.no_grad():
        images = vae.decode(latent / 0.18215).sample
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        grid = make_grid(images, nrow=int(np.ceil(np.sqrt(images.shape[0]))), padding=2)
        return Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


@torch.no_grad()
def generate_process_gif(
    checkpoint_path=None,
    output_gif=None,
    num_steps=50,
    batch_size=4,
    cfg_scale=4.0,
    frame_duration_ms=100,
    final_pause_sec=2.0
):
    current_dir = os.path.dirname(__file__)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(current_dir, "dit_celeba10.pt")
    if output_gif is None:
        output_gif = os.path.join(current_dir, "generation_process.gif")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    # Load DiT Model (prefer EMA weights)
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 10)
    model = DiT(
        input_size=ckpt["input_size"], patch_size=ckpt["patch_size"], in_channels=ckpt["in_channels"],
        hidden_size=ckpt.get("hidden_size", 384), depth=ckpt.get("depth", 8),
        num_heads=ckpt.get("num_heads", 12), num_classes=num_classes
    ).to(device)

    state_dict = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    vae = load_vae(device)

    # Class labels for batch generation
    y_cond = torch.tensor([i % num_classes for i in range(batch_size)], device=device)
    y_uncond = torch.full((batch_size,), fill_value=num_classes, device=device)

    # Euler integration ODE with Classifier-Free Guidance (CFG): t = 0.0 -> 1.0
    torch.manual_seed(42)
    x_t = torch.randn(batch_size, 4, 32, 32, device=device)
    dt = 1.0 / num_steps
    frames = [latent_to_pil(x_t, vae)]

    print(f"[Inference] Running ODE integration with CFG (scale={cfg_scale}): t=0.0 -> 1.0 ({num_steps} steps)...")
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((batch_size,), fill_value=t_val * 1000.0, device=device)

        v_cond = model(x_t, t_tensor, y_cond)
        v_uncond = model(x_t, t_tensor, y_uncond)

        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        x_t = x_t + dt * v_pred

        frames.append(latent_to_pil(x_t, vae))

    pause_frames = int(final_pause_sec * 10)
    frames += [frames[-1]] * pause_frames

    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=frame_duration_ms, loop=0)
    print(f"[Inference] Saved CFG generation animation to '{output_gif}'.")


if __name__ == "__main__":
    generate_process_gif()

