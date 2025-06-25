#!/usr/bin/env python3
# hgat_90.py – HGAT-90 for MalNet-Tiny  (RTX 4080 friendly)
# =========================================================
# Ví dụ chạy chắc chắn không OOM:
#   python hgat_90.py --device cuda --epochs 200 --batch_size 16 --hidden 128 --heads 8
#
# Cờ hữu ích:
#   --ckpt       bật gradient-checkpoint (tiết kiệm ~30 % VRAM, chậm ~1.3×)
#   --patience   early-stop (mặc định 10)

import argparse, math, torch
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader   import DataLoader
from torch_geometric.utils    import degree
from torch_geometric.nn       import GATv2Conv, global_mean_pool, LayerNorm
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

# ---------- node-feature (3-dim) ----------
def build_node_feature(g):
    if g.x is not None and g.x.size(1) >= 3:
        return
    din  = degree(g.edge_index[1], g.num_nodes).unsqueeze(1)
    dout = degree(g.edge_index[0], g.num_nodes).unsqueeze(1)
    size = torch.full((g.num_nodes, 1), math.log(g.num_nodes + 1e-6))
    g.x = torch.cat([din, dout, size], 1).float()

# ---------- sanitize dataset ----------
def sanitize(ds):
    out = []
    for g in ds:
        if g.num_nodes == 0:
            continue
        # ⬇︎ thêm alias nếu thiếu
        if not hasattr(g, "y_family"):
            g.y_family = getattr(g, "y", torch.zeros(1, dtype=torch.long))
        if not hasattr(g, "y_type"):
            g.y_type   = getattr(g, "type",
                          getattr(g, "type_id",
                                  torch.zeros(1, dtype=torch.long)))
        build_node_feature(g)
        out.append(g)
    return out

# ---------- HGAT-90 model ----------
class HGAT90(nn.Module):
    def __init__(self, in_dim, n_fam, n_type,
                 hidden=256, heads=16, layers=10,
                 type_emb=16, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.type_emb = nn.Embedding(n_type, type_emb)
        self.convs, self.lns, self.res = nn.ModuleList(), nn.ModuleList(), []
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads, dropout=0.3))
            self.lns .append(LayerNorm(hidden * heads))
            self.res .append(last == hidden*heads)
            last = hidden * heads
        self.cls_fam  = nn.Linear(last + type_emb, n_fam)
        self.cls_type = nn.Linear(last + type_emb, n_type)

    def _block(self, conv, ln, skip, h, edge):
        z = torch.relu(conv(h, edge))
        z = ln(z)
        return z + h if skip else z

    def forward(self, x, edge, batch, y_type):
        h = x
        for conv, ln, skip in zip(self.convs, self.lns, self.res):
            if self.training and self.ckpt:
                h = checkpoint(self._block, conv, ln, skip, h, edge)
            else:
                h = self._block(conv, ln, skip, h, edge)
        g = global_mean_pool(h, batch)
        g = torch.cat([g, self.type_emb(y_type)], 1)
        return self.cls_fam(g), self.cls_type(g)

# ---------- train / eval ----------
def train_epoch(model, loader, opt, scaler, sched, dev,
                alpha=0.3, use_amp=True):
    model.train(); total = 0
    for d in loader:
        d = d.to(dev)
        opt.zero_grad()
        ctx = torch.amp.autocast(device_type='cuda') if use_amp else nullcontext()
        with ctx:
            yf, yt = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
            loss = nn.CrossEntropyLoss()(yf, d.y_family.squeeze()) + \
                   alpha * nn.CrossEntropyLoss()(yt, d.y_type.squeeze())
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        total += loss.item() * d.num_graphs
    sched.step()
    return total / len(loader.dataset)

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval(); corr = tot = 0
    for d in loader:
        d = d.to(dev)
        pred, _ = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
        corr += (pred.argmax(1) == d.y_family.squeeze()).sum().item()
        tot  += d.num_graphs
    return corr / tot

# ---------- main ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/malnet_tiny')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--heads',  type=int, default=16)
    p.add_argument('--ckpt', action='store_true')
    p.add_argument('--patience', type=int, default=10)
    return p.parse_args()

def main():
    args = get_args()
    dev  = torch.device(args.device)
    use_amp = (dev.type == 'cuda')

    print('⏬ Loading MalNet-Tiny split …')
    train_ds = sanitize(MalNetTiny(args.root, split='train'))
    val_ds   = sanitize(MalNetTiny(args.root, split='val'))

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp)
    vl = DataLoader(val_ds,   batch_size=args.batch_size, pin_memory=use_amp)

    n_fam  = MalNetTiny(args.root).num_classes
    n_type = len({g.y_type.item() for g in train_ds})

    model  = HGAT90(3, n_fam, n_type,
                    hidden=args.hidden, heads=args.heads, ckpt=args.ckpt).to(dev)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if use_amp else None

    best, wait = 0, 0
    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, tl, opt, scaler, sched, dev, use_amp=use_amp)
        acc  = eval_acc (model, vl, dev)
        print(f'[E{ep:03d}] loss={loss:.4f} | val_acc={acc:.3f}')
        if acc > best:
            best, wait = acc, 0
            torch.save(model.state_dict(), 'best_hgat90.pt')
        else:
            wait += 1
            if wait == args.patience:
                print('⇡ Early-stopped.'); break
    print('✔ Best val-accuracy:', best)

if __name__ == '__main__':
    main()

