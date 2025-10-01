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
from add_noise import make_data_loader


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
#   Simple Generator (Conv, 28x28)
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


# ----------------------------
#   C2N Dependent Generator Pipe for 28x28
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, n_ch_in, ksize=3, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_ch_in, n_ch_in, ksize,
                      padding=(ksize // 2), bias=bias, padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv2d(n_ch_in, n_ch_in, ksize,
                      padding=(ksize // 2), bias=bias, padding_mode='reflect')
        )
    def forward(self, x):
        return x + self.body(x)

class C2N_MNIST_G(nn.Module):
    """
    Signal-dependent-only generator for MNIST (1x28x28).
    x -> (feature extractor w/ r-map) -> [mu, sigma] -> rsample -> 3x3 ResBlocks -> 1x1 conv -> noise
    y_hat = x + noise
    """
    def __init__(self, n_ch_in=1, # input channels
                n_ch_out=1, # output channels
                n_r=8, # length of r vector
                n_ch_unit=32,  # number of base channels
                n_ext=3, # number of residual blocks in feature extractor
                n_block_dep=3, # number of residual blocks in dependent module
                eps=1e-3):
        
        super().__init__()
        self.n_r = n_r
        self.n_ch_unit = n_ch_unit
        self.eps = eps

        # Feature extractor (make μ and σ features)
        self.ext_head = nn.Sequential(
            nn.Conv2d(n_ch_in, n_ch_unit, 3, padding=1, bias=True, padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv2d(n_ch_unit, 2 * n_ch_unit, 3, padding=1, bias=True, padding_mode='reflect'),  # -> 2*U
            nn.PReLU() # TODO check if this activation is needed/ should be there
        )
        # Merge r-map (broadcasted to HxW) with features, then mix back to 2*U
        self.ext_merge = nn.Sequential(
            nn.Conv2d(2 * n_ch_unit + n_r, 2 * n_ch_unit, 3, padding=1, bias=True, padding_mode='reflect'),
            nn.PReLU(),
        )

        # Residual trunk over 2*U channels
        self.ext = nn.Sequential(*[ResBlock(2 * n_ch_unit, ksize=3, bias=True) for _ in range(n_ext)])

        # Dependent refinement (ONLY 3x3, no 1x1 branch since MNIST has 1 channel)
        self.dep_refine = nn.Sequential(*[ResBlock(n_ch_unit, ksize=3, bias=False) for _ in range(n_block_dep)])

        # Tail to project noise-features (U) -> image channels (1)
        self.tail = nn.Conv2d(n_ch_unit, n_ch_out, 1, padding=0, bias=True, padding_mode='reflect')

    def forward(self, x, r_vector=None, return_aux=False):
        """
        x: (N,1,28,28)
        r_vector: (N, n_r) optional conditioning; if None, sampled ~ N(0,1)
        """
        N, C, H, W = x.shape

        # r-map (N, n_r, H, W)
        if r_vector is None:
            r_vector = torch.randn(N, self.n_r, device=x.device)
        r_map = r_vector.unsqueeze(-1).unsqueeze(-1).expand(N, self.n_r, H, W).detach()

        # Feature extractor
        feat = self.ext_head(x)                              # (N, 2U, H, W)
        feat = self.ext_merge(torch.cat([feat, r_map], dim=1))  # (N, 2U, H, W)
        feat = self.ext(feat)                                # (N, 2U, H, W)

        # Split into μ and σ channels (each U)
        mu     = feat[:, :self.n_ch_unit,  :, :]             # (N, U, H, W)
        sigma0 = feat[:,  self.n_ch_unit:, :, :]             # (N, U, H, W)
        sigma  = F.relu(sigma0) + self.eps                   # ensure positive std

        # Heteroscedastic sampling (reparameterized)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        dep_feat = dist.rsample().to(x.device)               # (N, U, H, W)

        # Refine with 3x3 ResBlocks
        dep_feat = self.dep_refine(dep_feat)                 # (N, U, H, W)

        # Project to 1 channel noise and add
        noise = self.tail(dep_feat)                          # (N, 1, H, W)
        y_hat = x + noise

        if return_aux:
            return y_hat, {"mu": mu, "sigma": sigma, "noise": noise}
        return y_hat


# ----------------------------
#   Critic for (B, 1, 28, 28)
# ----------------------------
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

# def save_samples(G: Generator, step: int, device: torch.device, z_dim: int,
#                  batch_size: int = 64, outdir: str = "samples"):
#     """
#     Save 10 generated samples individually as PNGs.
#     - batch_size: how many latent vectors to sample from z_dim
#     - only the first 10 images are saved
#     """
#     os.makedirs(outdir, exist_ok=True)
#     was_training = G.training
#     G.eval()
#     with torch.no_grad():
#         z = torch.randn(batch_size, z_dim, device=device)
#         imgs = G(z).cpu()
#         imgs = (imgs + 1) / 2.0  # denormalize to [0,1]

#         for i in range(10):
#             filename = os.path.join(outdir, f"sample_0{i}.png")
#             vutils.save_image(imgs[i], filename)
#     if was_training:
#         G.train()

def save_samples(G, step: int, device: torch.device,
                 clean_batch: torch.Tensor,
                 outdir: str = "samples", num_save: int = 10):
    """
    Save generated noisy samples given a batch of clean images.
    - clean_batch: (B,1,28,28) batch of clean MNIST images
    - num_save: how many images from the batch to save
    """
    os.makedirs(outdir, exist_ok=True)
    was_training = G.training
    G.eval()
    with torch.no_grad():
        clean_batch = clean_batch.to(device)
        noisy_batch = G(clean_batch).cpu()
        # noisy_batch = (noisy_batch + 1) / 2.0  # [0,1] for saving

        for i in range(min(num_save, noisy_batch.size(0))):
            filename = os.path.join(outdir, f"sample_{i:02d}_step{step}.png")
            vutils.save_image(noisy_batch[i], filename)

    if was_training:
        G.train()

# def get_loader(batch_size: int) -> DataLoader:
#     tfm = transforms.Compose([
#         transforms.ToTensor(),                   # [0,1]
#         transforms.Lambda(lambda x: x * 2 - 1), # [-1,1] for Tanh output
#     ])
#     ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
#     # return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

def get_loader(batch_size: int, pt_path) -> DataLoader:
    
    return make_data_loader(pt_path, batch_size, shuffle= True, num_workers= 2)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    z_dim = args.z_dim
    batch_size = args.batch_size
    critic_steps = args.critic_steps
    lambda_gp = args.lambda_gp
    total_steps = args.total_steps
    pt_path = args.pt_path

    # Data
    loader = get_loader(pt_path, batch_size)
    data_iter = iter(loader)

    # Models
    # G = Generator(z_dim=z_dim, img_ch=1).to(device)
    G = C2N_MNIST_G(n_ch_in=1, # input channels
                n_ch_out=1, # output channels
                n_r=8, # length of r vector
                n_ch_unit=32,  # number of base channels
                n_ext=3, # number of residual blocks in feature extractor
                n_block_dep=3, # number of residual blocks in dependent module
                eps=1e-3)
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
                # real, _ = next(data_iter)
                clean, noisy, _, _, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                # real, _ = next(data_iter)
                clean, noisy, _, _, _ = next(data_iter)

            # real = real.to(device)  # (B,1,28,28)
            noisy = noisy.to(device)  # (B,1,28,28)
            # bsz = real.size(0)
            # bsz = noisy.size(0) # no need for this

            clean = clean.to(device)

            # z = torch.randn(bsz, z_dim, device=device)
            fake = G(clean).detach()  # stop grad into G

            d_opt.zero_grad(set_to_none=True)

            d_real = D(noisy).mean()
            d_fake = D(fake).mean()

            gp = gradient_penalty(D, real=noisy, fake=fake, device=device) # noisy is real data and fake is output of G

            d_loss = d_fake - d_real + lambda_gp * gp
            d_loss.backward()
            d_opt.step()

        # -----------------------------
        #  Generator update (1 step)
        # -----------------------------
        for p in D.parameters(): p.requires_grad_(False)
        for p in G.parameters(): p.requires_grad_(True)

        # z = torch.randn(batch_size, z_dim, device=device)
        clean, noisy, _, _, _ = next(data_iter)  # fetch another batch
        clean = clean.to(device)
        noisy = noisy.to(device)

        g_opt.zero_grad(set_to_none=True)

        # fake = G(z)
        fake = G(clean)
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
    clean_batch, _, _, _, _ = next(iter(loader))
    save_samples(G, step, device, clean_batch, outdir=args.samples_dir)


def parse_args():
    p = argparse.ArgumentParser("c2n on MNIST (28x28)")
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
    p.add_argument("--pt_path", type=str, default="./mnist_noisy_pairs_train.pt"),
    p.add_argument("--run_dir", type=str, default="c2n_mnist_1",
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
