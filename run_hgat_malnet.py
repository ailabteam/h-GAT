#!/usr/bin/env python3
# run_hgat_malnet.py  –  download 1 chunk, build dataset & train HGAT
# ================================================================

import argparse, urllib.request, gzip, tarfile, subprocess, os
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

# -------------------- download & extract -------------------- #
def download_chunk(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[download] skip {dst.name}")
    else:
        print(f"[download] {url} → {dst}")
        urllib.request.urlretrieve(url, dst)

# ---------- thay thế toàn bộ hàm extract_partial ----------

def extract_partial(archive: Path, out_dir: Path):
    """
    Stream-extract .tar  *hoặc* .tar.gz, chịu được archive bị cắt.
    Thử gzip trước; nếu tarfile.ReadError thì rơi về tar thường.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def open_stream(gz: bool):
        return (gzip.open(archive, "rb") if gz else archive.open("rb"),
                "r|gz" if gz else "r|",
                "gzip" if gz else "tar")

    # -- thử gzip --
    fileobj, mode, kind = open_stream(True)
    try:
        tf = tarfile.open(fileobj=fileobj, mode=mode)
        tf.members = []          # tránh giữ reference
        fileobj.seek(0)          # ok → quay về đầu
    except (tarfile.ReadError, OSError):
        fileobj.close()
        fileobj, mode, kind = open_stream(False)   # fallback
        print("[extract] fallback → plain tar")

    print(f"[extract] streaming ({kind}) …")
    n = 0
    try:
        with tarfile.open(fileobj=fileobj, mode=mode) as tf:
            for ti in tf:
                try:
                    tf.extract(ti, path=out_dir)
                    n += 1
                except (tarfile.ExtractError, OSError):
                    continue
    except EOFError:
        print("  [warn] EOF – archive truncated (normal).")
    finally:
        fileobj.close()

    print(f"[extract] extracted {n:,} members")

# -------------------- build processed/data.pt -------------------- #
def load_edges(fp: Path):
    arr = []
    with fp.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 2:
                continue
            try:
                arr.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return None if not arr else np.asarray(arr, dtype=np.int64)

def build_pt(graph_root: Path, out_pt: Path):
    edge_files = list(graph_root.rglob("*.edgelist"))
    print(f"[prepare] {len(edge_files):,} edgelist files → Data list")
    fam2id, type2id, data_list = {}, {}, []
    for fp in tqdm(edge_files, unit="file"):
        gtype, gfamily = fp.parts[-3], fp.parts[-2]
        tid = type2id.setdefault(gtype, len(type2id))
        fid = fam2id.setdefault(gfamily, len(fam2id))

        edges = load_edges(fp)
        if edges is None:    # đồ thị rỗng hoặc lỗi
            continue
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        num_nodes  = int(edge_index.max()) + 1
        deg = torch.bincount(edge_index.reshape(-1),
                             minlength=num_nodes).unsqueeze(1).float()

        data_list.append(Data(x=deg,
                              edge_index=edge_index,
                              y_family=torch.tensor([fid]),
                              y_type=torch.tensor([tid])))

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, out_pt)
    print(f"[prepare] saved {len(data_list):,} graphs → {out_pt.relative_to(out_pt.parents[1])}")

# -------------------- HGAT model -------------------- #
class HGAT(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=64, heads=4):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden, heads=heads, dropout=0.2)
        self.g2 = GATConv(hidden*heads, hidden, heads=heads, dropout=0.2)
        self.cls = nn.Linear(hidden*heads, n_cls)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.g1(x, edge_index))
        x = torch.relu(self.g2(x, edge_index))
        hg = global_mean_pool(x, batch)
        return self.cls(hg)

# -------------------- train helpers -------------------- #
def train_one(model, loader, opt, device):
    model.train(); tot = 0
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        out = model(d.x, d.edge_index, d.batch)
        loss = nn.CrossEntropyLoss()(out, d.y_family.squeeze())
        loss.backward(); opt.step()
        tot += loss.item() * d.num_graphs
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); correct = total = 0
    for d in loader:
        d = d.to(device)
        pred = model(d.x, d.edge_index, d.batch).argmax(1)
        correct += (pred == d.y_family.squeeze()).sum().item()
        total   += d.num_graphs
    return correct / total

# -------------------- main -------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/malnet_sample1")
    p.add_argument("--base_url", default="http://malnet.cc.gatech.edu/graph-data/full-data-as-1GB/malnet-graph")
    p.add_argument("--chunk_idx", type=int, default=0, help="0 → malnet-graph00")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.data_root)
    raw_dir = root / "raw"
    processed_pt = root / "processed" / "data.pt"

    if not processed_pt.exists():
        # ------- download & extract (1 chunk) -------
        chunk_url = f"{args.base_url}{args.chunk_idx:02d}"
        chunk_file = raw_dir / f"malnet-graph{args.chunk_idx:02d}"
        download_chunk(chunk_url, chunk_file)
        tar_path = raw_dir / "malnet.tar"
        if not tar_path.exists():
            chunk_file.replace(tar_path)   # rename để dễ xử lý
        extract_dir = root / "extracted"
        extract_partial(tar_path, extract_dir)
        build_pt(extract_dir / "malnet-graphs", processed_pt)

    # ------- dataset as list[Data] -------
    data_list = torch.load(processed_pt)
    n_train   = int(0.8 * len(data_list))
    train_ds, val_ds = torch.utils.data.random_split(data_list, [n_train, len(data_list)-n_train])
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=args.batch_size)

    # ------- model & training -------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cls  = len({d.y_family.item() for d in data_list})
    model  = HGAT(1, n_cls).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, args.epochs+1):
        loss = train_one(model, tl, opt, device)
        acc  = eval_acc(model, vl, device)
        print(f"[E{ep:02d}] loss={loss:.4f} | val_acc={acc:.3f}")

if __name__ == "__main__":
    main()

