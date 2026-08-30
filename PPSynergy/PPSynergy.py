import torch
import torch.nn as nn
import torch.nn.functional as F
class NoisePredictor(nn.Module):
    def __init__(self, in_channels, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, in_channels)
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(-1) / 10
        t_embed = self.time_mlp(t)
        t_embed = t_embed[..., None, None]
        return self.net(x + t_embed)
    
class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, channels=3, timesteps=10,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.channels = channels
        self.image_size = image_size

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

    # q(x_t | x_0) - 
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_start + sqrt_one * noise

    # diffusion loss
    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        predicted_noise = self.model(x_noisy, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.betas[t].to(x.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(x.device)
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t]).to(x.device)

        model_mean = sqrt_recip_alpha * (
            x - betas_t / sqrt_one_minus_alpha * self.model(x, torch.tensor([t], device=x.device))
        )

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        return model_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, batch_size):
    
        x = torch.randn(batch_size, self.channels, self.image_size[0], self.image_size[1]).cuda()

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)

        return x
    
class PPSynergyNet(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.noise_model = NoisePredictor(3)
        self.diffusion = GaussianDiffusion(self.noise_model, image_size=image_size, timesteps=10)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.drop=nn.Dropout(0.2)

        self.fc1 = nn.Linear(16 * 93 * 46, 300)  
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # =========================
        # diffusion part
        # =========================
        t = torch.randint(
            0,
            self.diffusion.timesteps,
            (x.size(0),),
            device=x.device
        ).long()

        diffusion_loss = self.diffusion.p_losses(x, t)

        # forward diffusion
        noise = torch.randn_like(x)
        x_noisy = self.diffusion.q_sample(x, t, noise)
        predicted_noise = self.noise_model(x_noisy, t)
        x_denoised = x_noisy - predicted_noise
        # =========================
        # CNN classification 
        # =========================
        x = self.pool(F.relu(self.conv1(x_denoised)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.drop(x)
        x = self.fc3(x)
        return x, diffusion_loss
