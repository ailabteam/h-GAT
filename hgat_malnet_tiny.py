#!/usr/bin/env python3
# hgat_malnet_tiny.py  –  HGAT train trên MalNet-Tiny (family)
# ============================================================

import argparse, torch
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader   import DataLoader
from torch_geometric.utils    import degree
from torch_geometric.nn       import GATConv, global_mean_pool

# ---------- HGAT model ---------- #
class HGAT(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=64, heads=4, layers=2):
        super().__init__()
        self.gats = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.gats.append(GATConv(last, hidden, heads=heads, dropout=0.2))
            last = hidden * heads
        self.cls = nn.Linear(last, n_cls)

    def forward(self, x, edge_index, batch):
        for gat in self.gats:
            x = torch.relu(gat(x, edge_index))
        return self.cls(global_mean_pool(x, batch))

# ---------- utils ---------- #
def sanitize(dataset):
    """Thêm feature degree; bỏ graph không nút."""
    clean = []
    drop  = 0
    for g in dataset:
        if g.num_nodes == 0:           # skip completely empty graph
            drop += 1
            continue
        if g.x is None:
            if g.edge_index.numel() == 0:
                g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
            else:
                g.x = degree(g.edge_index[0], g.num_nodes).unsqueeze(1).float()
        clean.append(g)
    print(f"🧹 Dropped {drop} empty graphs; kept {len(clean)}")
    return clean

def split(ds, ratio=0.8, seed=42):
    torch.manual_seed(seed)
    n = int(ratio * len(ds))
    return torch.utils.data.random_split(ds, [n, len(ds)-n])

def train_epoch(model, loader, opt, device):
    model.train(); tot = 0
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(d.x, d.edge_index, d.batch),
                                     d.y.squeeze())
        loss.backward(); opt.step()
        tot += loss.item() * d.num_graphs
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); corr = tot = 0
    for d in loader:
        d = d.to(device)
        pred = model(d.x, d.edge_index, d.batch).argmax(1)
        corr += (pred == d.y.squeeze()).sum().item()
        tot  += d.num_graphs
    return corr / tot

# ---------- main ---------- #
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/malnet_tiny")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads",  type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    return p.parse_args()

def main():
    args = parse()
    print("⏬ Loading MalNet-Tiny …")
    raw_ds = MalNetTiny(root=args.root)          # family task mặc định
    ds     = sanitize(raw_ds)                    # lọc & add feature

    train_ds, val_ds = split(ds)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = HGAT(1, raw_ds.num_classes,
                  hidden=args.hidden, heads=args.heads, layers=args.layers).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, tl, opt, device)
        acc  = eval_acc (model, vl, device)
        print(f"[E{ep:02d}] loss={loss:.4f} | val_acc={acc:.3f}")

if __name__ == "__main__":
    main()

