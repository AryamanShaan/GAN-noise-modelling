import os
import math
import random
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


# ----------------------------
#   Reproducibility helpers
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
#   Models (Conv, 28x28)
# ----------------------------
class Generator(nn.Module):
    """
    z (B, z_dim) --> (B, 1, 28, 28)
    Architecture: project to 128x7x7, then upsample by ConvT to 14x14 and 28x28
    """
    def __init__(self, z_dim: int = 100, img_ch: int = 1):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)

        self.deconv = nn.Sequential(
            # 128 x 7 x 7  ->  64 x 14 x 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 14 x 14 ->  32 x 28 x 28
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 x 28 x 28 ->   1 x 28 x 28
            nn.Conv2d(32, img_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)
        return self.deconv(x)


class Critic(nn.Module):
    """
    (B, 1, 28, 28) -> scalar score (B, 1, 1, 1)
    No normalization layers per WGAN-GP recommendations.
    """
    def __init__(self, img_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # 1 x 28 x 28 -> 32 x 14 x 14
            nn.Conv2d(img_ch, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 14 x 14 -> 64 x 7 x 7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 7 x 7 -> 128 x 7 x 7 (keep size, add capacity)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 7 x 7 -> 1 x 1 x 1
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # (B,)


# ----------------------------
#   Gradient Penalty
# ----------------------------
def gradient_penalty(critic: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    WGAN-GP gradient penalty on interpolations between real and fake.
    Returns the scalar penalty term (without lambda).
    """
    bsz = real.size(0)
    # Sample interpolation coefficients and expand to image shape
    epsilon = torch.rand(bsz, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    # Critic scores
    scores = critic(interpolated)  # (B,)
    ones = torch.ones_like(scores, device=device)

    # Compute gradients w.r.t. interpolated inputs
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # shape: (B, C, H, W)

    grads = grads.view(bsz, -1)
    grad_norm = grads.norm(2, dim=1)  # (B,)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty


# ----------------------------
#   Training Loop
# ----------------------------

def save_samples(G: Generator, step: int, device: torch.device, z_dim: int,
                 batch_size: int = 64, outdir: str = "samples"):
    """
    Save 10 generated samples individually as PNGs.
    - batch_size: how many latent vectors to sample from z_dim
    - only the first 10 images are saved
    """
    os.makedirs(outdir, exist_ok=True)
    was_training = G.training
    G.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim, device=device)
        imgs = G(z).cpu()
        imgs = (imgs + 1) / 2.0  # denormalize to [0,1]

        for i in range(10):
            filename = os.path.join(outdir, f"sample_0{i}.png")
            vutils.save_image(imgs[i], filename)
    if was_training:
        G.train()

def get_loader(batch_size: int) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),                   # [0,1]
        transforms.Lambda(lambda x: x * 2 - 1), # [-1,1] for Tanh output
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    z_dim = args.z_dim
    batch_size = args.batch_size
    critic_steps = args.critic_steps
    lambda_gp = args.lambda_gp
    total_steps = args.total_steps

    # Data
    loader = get_loader(batch_size)
    data_iter = iter(loader)

    # Models
    G = Generator(z_dim=z_dim, img_ch=1).to(device)
    D = Critic(img_ch=1).to(device)

    # Optims (WGAN-GP rec: betas = (0.0, 0.9))
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    step = 0
    G.train()
    D.train()

    while step < total_steps:
        # -----------------------------
        #  Critic updates (k steps)
        # -----------------------------
        for p in D.parameters(): p.requires_grad_(True)
        for p in G.parameters(): p.requires_grad_(False)

        for _ in range(critic_steps):
            try:
                real, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real, _ = next(data_iter)

            real = real.to(device)  # (B,1,28,28)
            bsz = real.size(0)

            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()  # stop grad into G

            d_opt.zero_grad(set_to_none=True)

            d_real = D(real).mean()
            d_fake = D(fake).mean()

            gp = gradient_penalty(D, real, fake, device)

            d_loss = d_fake - d_real + lambda_gp * gp
            d_loss.backward()
            d_opt.step()

        # -----------------------------
        #  Generator update (1 step)
        # -----------------------------
        for p in D.parameters(): p.requires_grad_(False)
        for p in G.parameters(): p.requires_grad_(True)

        z = torch.randn(batch_size, z_dim, device=device)

        g_opt.zero_grad(set_to_none=True)

        fake = G(z)
        # maximize D(fake)  <=> minimize -D(fake)
        g_loss = - D(fake).mean()
        g_loss.backward()
        g_opt.step()

        step += 1

        if step % args.log_every == 0:
            wasserstein = (d_real - d_fake).item()
            print(f"[step {step:6d}] D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}  W-dist≈{wasserstein:.4f}  GP={gp.item():.4f}")

        # if step % args.sample_every == 0:
        #     save_samples(G, step, device, z_dim, outdir=args.samples_dir)

        if step == total_steps-1:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), os.path.join(args.ckpt_dir, f"G_step{step}.pt"))
            torch.save(D.state_dict(), os.path.join(args.ckpt_dir, f"D_step{step}.pt"))

    # final samples
    save_samples(G, step, device, z_dim, batch_size, outdir=args.samples_dir)


def parse_args():
    p = argparse.ArgumentParser("WGAN-GP on MNIST (28x28)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--critic_steps", type=int, default=5)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--log_every", type=int, default=100)
    # p.add_argument("--sample_every", type=int, default=1000)
    # p.add_argument("--ckpt_every", type=int, default=5000)
    # p.add_argument("--samples_dir", type=str, default="samples")
    # p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--run_dir", type=str, default="wgan-gp_1",
                   help="subdirectory name for this run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Subdirectories
    args.samples_dir = os.path.join(args.run_dir, "samples")
    args.ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train(args)
