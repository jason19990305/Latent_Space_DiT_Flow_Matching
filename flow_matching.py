import torch


class FlowMatcher:
    """
    Optimal Transport Flow Matching (OT-CFM)
    - t = 0: Noise x_0 ~ N(0, I)
    - t = 1: Data  x_1 (VAE Latent)
    - Path:  x_t = (1 - t) * x_0 + t * x_1
    - Target velocity: v_t = x_1 - x_0
    """

    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps

    def compute_loss(self, model, x_1, y):
        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0],), device=x_1.device)
        x_t = (1 - t.view(-1, 1, 1, 1)) * x_0 + t.view(-1, 1, 1, 1) * x_1
        v_target = x_1 - x_0
        v_pred = model(x_t, t * self.num_train_timesteps, y)
        return torch.mean((v_pred - v_target) ** 2)

    @torch.no_grad()
    def euler_step(self, model, x_t, t_val, dt, y):
        """Single Euler ODE step: x_{t+dt} = x_t + dt * v_pred"""
        t = torch.full((x_t.shape[0],), fill_value=t_val * self.num_train_timesteps, device=x_t.device)
        return x_t + dt * model(x_t, t, y)
