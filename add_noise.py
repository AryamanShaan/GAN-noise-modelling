import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import math
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from typing import Literal, Optional


@torch.no_grad()
def add_heteroscedastic_gaussian(
    x,
    clamp=True
):
    # alpha_range=(0.1, 0.9)   # range for alpha (per pixel)
    # delta_range=(0.01, 0.09) # range for delta (per pixel)
    x = x.float()

    # sample per-pixel alpha and delta in the same shape as x
    # alpha = torch.empty_like(x).uniform_(alpha_range[0], alpha_range[1])
    # delta = torch.empty_like(x).uniform_(delta_range[0], delta_range[1])
    alpha = torch.full_like(x, 0.6)  
    delta = torch.full_like(x, 0.05)  

    # variance = alpha^2 * x + delta^2
    var = (alpha ** 2) * x + (delta ** 2)
    std = torch.sqrt(var)

    noise = torch.randn_like(x) * std
    y = x + noise

    if clamp:
        y = y.clamp(0.0, 1.0)

    return noise, y, x


@torch.no_grad()
def build_mnist_noisy_pairs(
    out_path: str = "mnist_noisy_pairs_train.pt",
    split: Literal["train","test"] = "train",
    root: str = "./data",
    batch_size: int = 2048,
    dtype: torch.dtype = torch.float32,   
    mismatch_pairs: bool = True,
    seed: int = 42
):
    """
    Creates {clean, noisy, clean_labels, noisy_labels, perm, meta} and saves to out_path.

    - clean/noisy are in [0,1], dtype (default float16)
    - If mismatch_pairs=True, 'noisy' & 'noisy_labels' are shuffled by a permutation 'perm'
      so (clean[i], noisy[i]) typically do NOT correspond to the same source image.
    """
    # device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    tfm = transforms.ToTensor()  # MNIST returns [0,1], shape (1,28,28)
    is_train = (split == "train")
    ds = datasets.MNIST(root=root, train=is_train, download=True, transform=tfm)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    clean_list, noisy_list, clabels_list, nlabels_list = [], [], [], []

    for x, y in loader:
        x = x.to(device)                    # [B,1,28,28], in [0,1]
        _, y_noisy, _ = add_heteroscedastic_gaussian(x, clamp=True)
        clean_list.append(x.to(dtype=dtype).cpu())
        noisy_list.append(y_noisy.to(dtype=dtype).cpu())
        clabels_list.append(y.cpu())
        nlabels_list.append(y.cpu())        # initially same labels as clean (pre-mismatch)

    clean = torch.cat(clean_list, dim=0)          # [N,1,28,28]
    noisy = torch.cat(noisy_list, dim=0)          # [N,1,28,28]
    clean_labels = torch.cat(clabels_list, dim=0) # [N]
    noisy_labels = torch.cat(nlabels_list, dim=0) # [N]
    N = clean.size(0)

    # Make mismatched pairs by shuffling the noisy side only
    if mismatch_pairs:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        noisy = noisy[perm]
        noisy_labels = noisy_labels[perm]
    else:
        perm = torch.arange(N)

    meta = {
        "split": split,
        "dtype": str(dtype).replace("torch.", ""),
        "mismatch_pairs": mismatch_pairs,
        "seed": seed,
        "num_examples": N,
        "note": "MNIST clean/noisy heteroscedastic (alpha=0.6, delta=0.05). [0,1] range."
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "clean": clean,                   # [N,1,28,28], float16
            "noisy": noisy,                   # [N,1,28,28], float16
            "clean_labels": clean_labels,     # [N]
            "noisy_labels": noisy_labels,     # [N]
            "perm": perm,                     # [N], how we shuffled noisy
            "meta": meta,
        },
        out_path,
    )
    print(f"Saved: {out_path} ({split=}, N={N}, mismatch={mismatch_pairs})")


class MnistCleanNoisyPairs(torch.utils.data.Dataset):
    """
    Yields a tuple:
        clean[i]:        FloatTensor (1,28,28) in [0,1]
        noisy[i]:        FloatTensor (1,28,28) in [0,1]
        clean_label[i]:  int
        noisy_label[i]:  int  (note: often different if mismatch_pairs=True)
        idx:             int (original index of the clean item)
    """
    def __init__(self, pt_path: str, load_to_device: Optional[torch.device] = None):
        blob = torch.load(pt_path, map_location="cpu")
        self.clean = blob["clean"]           # [N,1,28,28], float16
        self.noisy = blob["noisy"]
        self.clean_labels = blob["clean_labels"].long()
        self.noisy_labels = blob["noisy_labels"].long()
        self.perm = blob["perm"]
        self.meta = blob["meta"]

        if load_to_device is not None:
            # optionally move to GPU for ultra-fast training, if it fits
            self.clean = self.clean.to(load_to_device, non_blocking=True)
            self.noisy = self.noisy.to(load_to_device, non_blocking=True)
            self.clean_labels = self.clean_labels.to(load_to_device, non_blocking=True)
            self.noisy_labels = self.noisy_labels.to(load_to_device, non_blocking=True)

    def __len__(self):
        return self.clean.size(0)

    def __getitem__(self, i: int):
        return (
            self.clean[i],        # [1,28,28], float16/float
            self.noisy[i],        # [1,28,28], float16/float
            self.clean_labels[i], # int
            self.noisy_labels[i], # int
            i,                    # original clean index
        )

# Example DataLoader
def make_data_loader(pt_path: str, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
    ds = MnistCleanNoisyPairs(pt_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# assumes MnistCleanNoisyPairs class is already defined/imported
def save_random_pairs(pt_path: str, outdir: str = "investigate_data_samples", n: int = 10, seed: int = 42):
    os.makedirs(outdir, exist_ok=True)
    ds = MnistCleanNoisyPairs(pt_path)
    N = len(ds)

    # sample n unique indices reproducibly
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(N, generator=g)[:n].tolist()

    rows = []
    for k, i in enumerate(idxs):
        clean, noisy, y_clean, y_noisy, idx = ds[i]
        clean = clean.float()  # ensure float in [0,1]
        noisy = noisy.float()

        # make one row: [clean | noisy]
        row = make_grid([clean, noisy], nrow=2, padding=2)
        rows.append(row)

        # save per-pair image
        fn = f"{k:02d}_idx{idx}_yc{int(y_clean)}_yn{int(y_noisy)}.png"
        save_image(row, os.path.join(outdir, fn))

    # also save a single summary grid (10 rows stacked vertically)
    summary = make_grid(rows, nrow=1, padding=4)
    save_image(summary, os.path.join(outdir, f"summary_{n}pairs.png"))

    print(f"Saved {n} pairs to {outdir}")
    
def main():

    # build dataset
    # build_mnist_noisy_pairs(
    # out_path= "mnist_noisy_pairs_train.pt",
    # split = "train",
    # root = "./data",
    # batch_size= 2048,
    # dtype = torch.float32,   
    # mismatch_pairs = True,
    # seed = 42
    # )

    # inspect dataset
    save_random_pairs("mnist_noisy_pairs_train.pt", outdir="investigate_data_samples", n=10, seed=50)


if __name__ == "__main__":
    main()