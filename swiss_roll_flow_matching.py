"""
Swiss Roll Flow Matching (OT-CFM)
----------------------------------
  t = 0  :  Noise  x_0 ~ N(0, I)
  t = 1  :  Data   x_1 ~ Swiss Roll
  Path   :  x_t = (1 - t) * x_0 + t * x_1
  Target :  v_t = x_1 - x_0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import make_swiss_roll


# -- Model ---------------------------------------------------------------------

class VelocityNet(nn.Module):
    """MLP that predicts the velocity field v(x_t, t)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(1)], dim=1))


# -- Flow Matching -------------------------------------------------------------

class FlowMatcher:
    """
    Optimal Transport Flow Matching (OT-CFM)
    - t = 0: Noise x_0 ~ N(0, I)
    - t = 1: Data  x_1 (Swiss Roll)
    - Path:  x_t = (1 - t) * x_0 + t * x_1
    - Target velocity: v_t = x_1 - x_0
    """

    def compute_loss(self, model, x_1):
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t.unsqueeze(1)) * x_0 + t.unsqueeze(1) * x_1
        v_pred = model(x_t, t)
        return torch.mean((v_pred - (x_1 - x_0)) ** 2)


# -- Visualization -------------------------------------------------------------

def make_frame(pts, t):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.7, edgecolors='none')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--',  alpha=0.6)
    ax.set_title(f"Generating Swiss Roll via Flow Matching\nTime t = {t:.2f}", fontsize=11)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(frame[:, :, :3])

# -- Train & Inference ---------------------------------------------------------

def run(n_samples=2000, num_epochs=10000, lr=1e-3, output_gif='swiss_roll_flow.gif'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Swiss Roll: use x and z as 2D points, normalize
    data, colors = make_swiss_roll(n_samples=n_samples, noise=0.1)
    data = data[:, [0, 2]]
    data = (data - data.mean(0)) / data.std(0)
    x_1 = torch.tensor(data, dtype=torch.float32, device=device)

    model = VelocityNet().to(device)
    flow_matcher = FlowMatcher()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    print(f"Training for {num_epochs} epochs...")
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = flow_matcher.compute_loss(model, x_1)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == 1:
            print(f"Epoch [{epoch:05d}/{num_epochs}] | Loss: {loss.item():.6f}")

    # Inference: Euler ODE + save GIF
    print("Generating animation...")
    model.eval()
    num_gif_steps = 50
    x_t = torch.randn(500, 2, device=device)
    dt = 1.0 / num_gif_steps
    frames = []

    for step in range(num_gif_steps + 1):
        t_val = step * dt
        frames.append(make_frame(x_t.cpu().numpy(), t_val))
        if step < num_gif_steps:
            with torch.no_grad():
                t = torch.full((x_t.shape[0],), t_val, device=device)
                x_t = x_t + dt * model(x_t, t)

    frames += [frames[-1]] * 20
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"Saved animation to '{output_gif}'.")


if __name__ == '__main__':
    run()
