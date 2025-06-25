#!/usr/bin/env python3
# hgat_90.py – HGAT-90 + logging to file & console
# ===============================================

import argparse, math, torch, logging, datetime, os
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader   import DataLoader
from torch_geometric.utils    import degree
from torch_geometric.nn       import GATv2Conv, global_mean_pool, LayerNorm
from torch.utils.checkpoint   import checkpoint
from contextlib import nullcontext

# ---------- logging helper ----------
def init_logger(log_file: str = None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
    root = logging.getLogger(); root.setLevel(logging.INFO)
    sh   = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, mode='w'); fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.info

# ---------- node feature ----------
def build_node_feature(g):
    if g.x is not None and g.x.size(1) >= 3:
        return
    din  = degree(g.edge_index[1], g.num_nodes).unsqueeze(1)
    dout = degree(g.edge_index[0], g.num_nodes).unsqueeze(1)
    size = torch.full((g.num_nodes,1), math.log(g.num_nodes+1e-6))
    g.x  = torch.cat([din, dout, size], 1).float()

# ---------- sanitize ----------
def sanitize(ds):
    out = []
    for g in ds:
        if g.num_nodes == 0: continue
        g.y_family = getattr(g, "y_family",
                      getattr(g, "y", torch.zeros(1, dtype=torch.long)))
        g.y_type   = getattr(g, "y_type",
                      getattr(g, "type",
                              getattr(g, "type_id", torch.zeros(1, dtype=torch.long))))
        build_node_feature(g)
        out.append(g)
    return out

# ---------- HGAT ----------
class HGAT90(nn.Module):
    def __init__(self, in_dim, n_fam, n_type,
                 hidden=128, heads=8, layers=10,
                 type_emb=16, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.type_emb = nn.Embedding(n_type, type_emb)
        self.convs, self.lns, self.res = nn.ModuleList(), nn.ModuleList(), []
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads, dropout=0.3))
            self.lns .append(LayerNorm(hidden*heads))
            self.res .append(last == hidden*heads)
            last = hidden * heads
        self.cls_fam  = nn.Linear(last + type_emb, n_fam)
        self.cls_type = nn.Linear(last + type_emb, n_type)

    def _blk(self, conv, ln, skip, h, edge):
        z = torch.relu(conv(h, edge))
        z = ln(z)
        return z + h if skip else z

    def forward(self, x, edge, batch, y_type):
        h = x
        for conv, ln, skip in zip(self.convs, self.lns, self.res):
            h = (checkpoint(self._blk, conv, ln, skip, h, edge)
                 if self.training and self.ckpt
                 else self._blk(conv, ln, skip, h, edge))
        g = global_mean_pool(h, batch)
        g = torch.cat([g, self.type_emb(y_type)], 1)
        return self.cls_fam(g), self.cls_type(g)

# ---------- train / eval ----------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch(model, loader, opt, scaler, sched, dev, alpha=0.3, amp=True):
    model.train(); tot = 0
    for d in loader:
        d = d.to(dev); opt.zero_grad()
        ctx = torch.amp.autocast(device_type='cuda') if amp else nullcontext()
        with ctx:
            yf, yt = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
            loss = criterion(yf, d.y_family.squeeze()) + \
                   alpha * criterion(yt, d.y_type.squeeze())
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tot += loss.item() * d.num_graphs
    sched.step()
    return tot / len(loader.dataset)

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
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--heads',  type=int, default=8)
    p.add_argument('--layers', type=int, default=10)
    p.add_argument('--ckpt', action='store_true')
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--log_file', default=None,
                   help='path to log file; default runs/hgat_YYYYMMDD_HHMM.log')
    return p.parse_args()

def main():
    args = get_args()
    if args.log_file is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.log_file = f'runs/hgat_{ts}.log'
    log = init_logger(args.log_file)

    dev  = torch.device(args.device)
    use_amp = (dev.type == 'cuda')

    log('⏬ Loading MalNet-Tiny split …')
    train_ds = sanitize(MalNetTiny(args.root, split='train'))
    val_ds   = sanitize(MalNetTiny(args.root, split='val'))

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp)
    vl = DataLoader(val_ds,   batch_size=args.batch_size, pin_memory=use_amp)

    n_fam  = MalNetTiny(args.root).num_classes
    n_type = len({g.y_type.item() for g in train_ds})

    model = HGAT90(3, n_fam, n_type,
                   hidden=args.hidden, heads=args.heads,
                   layers=args.layers, ckpt=args.ckpt).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, 5)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs-5)
    sched= torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [5])
    scaler = torch.amp.GradScaler() if use_amp else None

    best, wait = 0, 0
    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, tl, opt, scaler, sched, dev, use_amp=use_amp)
        acc  = eval_acc(model, vl, dev)
        log(f'[E{ep:03d}] loss={loss:.4f} | val_acc={acc:.3f}')
        if acc > best:
            best, wait = acc, 0
            torch.save(model.state_dict(), 'best_hgat90.pt')
        else:
            wait += 1
            if wait == args.patience:
                log('⇡ Early-stopped.'); break
    log(f'✔ Best val-accuracy: {best:.3f}')
    log(f'Log saved to {args.log_file}')

if __name__ == '__main__':
    main()

