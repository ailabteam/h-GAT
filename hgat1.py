#!/usr/bin/env python3
"""hgat_90.py – HGAT‑90 for MalNet‑Tiny (family classification)
===============================================================
Target ≈ 90 % accuracy (paper baseline) on RTX 4080 (16 GB).
Key ingredients
---------------
• Node‑feature 3‑D (in‑deg, out‑deg, log|V|)  ➜ easy, cheap.  
• Deep **GATv2** (10×) 256 hidden, 16 heads, Residual + LayerNorm.  
• **Virtual‑node** token connects to all nodes for global context.  
• **TopKPooling** (ratio 0.5) generates super‑graph ➜ extra 3×GATv2.  
• **Type‑embedding** (5×16) concat at read‑out, **multi‑task** loss
  (family + 0.3·type) to respect hierarchy.  
• Mixed‑precision (AMP), AdamW + Cosine LR, Early‑Stopping (patience 20).

Run
---
```bash
python hgat_90.py --device cuda --epochs 200 --batch_size 64
```
Requires PyTorch >=2.2, PyG >=2.4 .
"""
import argparse, math, torch, random, warnings
from pathlib import Path
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GATv2Conv, LayerNorm, TopKPooling, global_mean_pool
warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------
# ---------- utility: node‑feature & sanitising --------------
# ------------------------------------------------------------

@torch.no_grad()
def build_feats(g):
    if g.num_nodes == 0:
        return False  # drop
    if g.x is not None and g.x.size(1) >= 3:
        return True   # already fine
    # degree based features
    indeg  = degree(g.edge_index[1], g.num_nodes).unsqueeze(1)
    outdeg = degree(g.edge_index[0], g.num_nodes).unsqueeze(1)
    size   = math.log(g.num_nodes + 1e-6)
    size_feat = torch.full((g.num_nodes, 1), size)
    g.x = torch.cat([indeg, outdeg, size_feat], dim=1).float()
    return True

@torch.no_grad()
def sanitize(ds):
    keep = []
    for g in ds:
        if build_feats(g):
            keep.append(g)
    return keep

# ------------------------------------------------------------
# -------------------- model blocks --------------------------
# ------------------------------------------------------------
class GATStack(nn.Module):
    """N layers of GATv2 with residual & LayerNorm"""
    def __init__(self, in_dim, hidden, heads, layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res   = []
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads, dropout=0.3))
            self.norms.append(LayerNorm(hidden * heads))
            self.res.append(last == hidden * heads)
            last = hidden * heads
        self.out_dim = last

    def forward(self, x, edge_index):
        out = x
        for conv, ln, skip in zip(self.convs, self.norms, self.res):
            h = torch.relu(conv(out, edge_index))
            h = ln(h)
            out = h + out if skip else h
        return out

class HGAT90(nn.Module):
    def __init__(self, in_dim, n_family, n_type, hidden=256, heads=16,
                 deep_layers=10, pool_ratio=0.5, type_emb_dim=16, top_layers=3):
        super().__init__()
        # virtual‑token embedding (shared) will be appended as node idx = num_nodes
        self.global_token = nn.Parameter(torch.zeros(1, in_dim))

        # deep local stack
        self.backbone = GATStack(in_dim, hidden, heads, deep_layers)
        # pooling
        self.pool = TopKPooling(self.backbone.out_dim, ratio=pool_ratio)
        # super‑graph stack
        self.top = GATStack(self.backbone.out_dim, hidden, heads, top_layers)

        # type embedding + classifiers
        self.type_emb = nn.Embedding(n_type, type_emb_dim)
        final_dim = self.top.out_dim + type_emb_dim
        self.cls_family = nn.Linear(final_dim, n_family)
        self.cls_type   = nn.Linear(final_dim,  n_type)

    def add_virtual(self, x, edge_index, batch):
        """Append global token per graph (shared param replicated)."""
        gs = torch.unique(batch)
        tok = self.global_token.expand(gs.numel(), -1)
        x = torch.cat([x, tok], dim=0)
        # connect token to every node of its graph
        offset = 0
        new_edges = []
        for g_id in gs:
            idx = torch.nonzero(batch == g_id, as_tuple=False).view(-1) + offset
            tok_idx = x.size(0) - gs.numel() + (g_id - gs[0])  # relative
            tok_vec = tok_idx.repeat(idx.numel())
            new_edges.append(torch.stack([idx, tok_vec], dim=0))
            new_edges.append(torch.stack([tok_vec, idx], dim=0))
            offset += 0  # batch contiguous
        edge_index = torch.cat([edge_index] + new_edges, dim=1)
        batch = torch.cat([batch, gs])
        return x, edge_index, batch

    def forward(self, x, edge_index, batch, type_id):
        # add virtual node edges
        x, edge_index, batch = self.add_virtual(x, edge_index, batch)
        # deep local
        x = self.backbone(x, edge_index)
        # hierarchical pooling
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        # super‑graph processing
        x = self.top(x, edge_index)
        g_emb = global_mean_pool(x, batch)
        g_emb = torch.cat([g_emb, self.type_emb(type_id)], dim=1)
        return self.cls_family(g_emb), self.cls_type(g_emb)

# ------------------------------------------------------------
# ----------------------- training loop ----------------------
# ------------------------------------------------------------
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); correct = total = 0
    for d in loader:
        d = d.to(device)
        pred, _ = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
        correct += (pred.argmax(1) == d.y_family.squeeze()).sum().item()
        total   += d.num_graphs
    return correct / total

scaler = torch.cuda.amp.GradScaler()

def train_epoch(model, loader, opt, sched, device, alpha):
    model.train(); tot = 0
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out_fam, out_type = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
            loss = nn.functional.cross_entropy(out_fam, d.y_family.squeeze()) \
                 + alpha * nn.functional.cross_entropy(out_type, d.y_type.squeeze())
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * d.num_graphs
    return tot / len(loader.dataset)

# ------------------------------------------------------------
# ----------------------- main entry -------------------------
# ------------------------------------------------------------

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/malnet_tiny')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--alpha', type=float, default=0.3, help='type loss weight')
    return p.parse_args()


def main():
    args = parse()
    print('⏬ load MalNet-Tiny official split…')
    train_raw = MalNetTiny(args.root, split='train')
    val_raw   = MalNetTiny(args.root, split='val')
    test_raw  = MalNetTiny(args.root, split='test')

    train_ds, val_ds, test_ds = map(sanitize, (train_raw, val_raw, test_raw))
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    n_family = train_raw.num_classes  # 47
    n_type   = len({g.y_type.item() for g in train_ds})

    model = HGAT90(in_dim=3, n_family=n_family, n_type=n_type).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, wait = 0, 0
    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, tl, opt, sched, args.device, args.alpha)
        acc  = eval_acc(model, vl, args.device)
        print(f'[E{ep:03d}] loss={loss:.4f} | val_acc={acc:.3f}')
        if acc > best:
            torch.save(model.state_dict(), 'best_hgat90.pt')
            best, wait = acc, 0
        else:
            wait += 1
            if wait == args.patience:
                print('◆ early‑stop'); break
    print('✔ best val_acc:', best)

if __name__ == '__main__':
    main()

