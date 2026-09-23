#!/usr/bin/env python3
"""
Leave-One-Design-Out (LODO) Cross-Validation for GNN timing predictor.
Runs 5 experiments — each time holds out one design, trains on the rest.
Also runs the fixed PicoRV32 test (the main paper result).
Saves: results/data/lodo_results.csv
       results/data/lodo_summary.csv
Usage: python run_gnn_lodo.py
"""
import json, csv, pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

ROOT    = Path(__file__).parent
GRAPH_D = ROOT / "ext5-gnn/graphs"
OUT_D   = ROOT / "results/data"
OUT_D.mkdir(parents=True, exist_ok=True)

EPOCHS      = 300
LR          = 1e-3
HIDDEN      = 64
NUM_LAYERS  = 3
RANDOM_SEED = 42

# ── Model (same as run_gnn_train.py) ─────────────────────────────────────────
class TimingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.mlp = Sequential(
            Linear(hidden * 2, hidden), ReLU(), Linear(hidden, 1))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.mlp(x).squeeze(-1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_graph(name):
    p = GRAPH_D / f"{name}.pt"
    g = torch.load(p, weights_only=False)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return g

def load_subgraphs(tag_prefix):
    """Load all subgraphs whose filename starts with tag_prefix."""
    graphs = []
    for p in sorted(GRAPH_D.glob(f"{tag_prefix}_s[0-9]*.pt")):
        g = torch.load(p, weights_only=False)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
        graphs.append(g)
    return graphs

def train_epoch(model, optimizer, graphs):
    model.train()
    total = 0.0
    for g in graphs:
        optimizer.zero_grad()
        pred = model(g.x, g.edge_index, g.batch)
        loss = F.mse_loss(pred, g.y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(graphs)

@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    preds, actuals = [], []
    for g in graphs:
        pred   = model(g.x, g.edge_index, g.batch).item()
        actual = g.y.item()
        preds.append(pred)
        actuals.append(actual)
    preds   = np.array(preds)
    actuals = np.array(actuals)
    mae  = float(np.mean(np.abs(preds - actuals)))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, r2, preds.tolist(), actuals.tolist()

def train_model(train_graphs, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    in_ch = train_graphs[0].x.shape[1]
    model = TimingGNN(in_ch, HIDDEN, NUM_LAYERS)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sch   = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_graphs)
        sch.step()
    return model

# ── Design registry ───────────────────────────────────────────────────────────
# Each entry: design_key → (full graph names, subgraph tag prefixes)
DESIGN_REGISTRY = {
    "GCD": {
        "full":     ["gcd", "gcd_5ns", "gcd_8ns", "gcd_10ns", "gcd_15ns"],
        "subgraph_prefixes": [],
    },
    "UART": {
        "full":     ["uart", "uart_4ns", "uart_4p5ns", "uart_6ns"],
        "subgraph_prefixes": [],
    },
    "AES-ORFS": {
        "full":     ["aes_orfs", "aes_orfs_9ns", "aes_orfs_10p5ns", "aes_orfs_12ns"],
        "subgraph_prefixes": ["aes_orfs_9ns", "aes_orfs_10ns", "aes_orfs_12ns"],
    },
    "SPI": {
        "full":     ["spi", "spi_1p2ns", "spi_1p8ns", "spi_2p5ns"],
        "subgraph_prefixes": [],
    },
    "AES": {
        "full":     ["aes"],
        "subgraph_prefixes": ["aes_31ns"],
    },
}

# Fixed test design (never used in any training run)
PICORV32_FULL = ["picorv32"]
PICORV32_SUBS = ["picorv32_13ns"]

def load_design_graphs(design_key):
    reg = DESIGN_REGISTRY[design_key]
    graphs = [load_graph(n) for n in reg["full"]]
    for prefix in reg["subgraph_prefixes"]:
        graphs.extend(load_subgraphs(prefix))
    return graphs

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)

    all_designs = list(DESIGN_REGISTRY.keys())
    detail_rows  = []
    summary_rows = []

    # ── LODO experiments ──────────────────────────────────────────────────────
    for held_out in all_designs:
        print(f"\n{'='*55}")
        print(f"LODO: held-out = {held_out}")
        train_designs = [d for d in all_designs if d != held_out]

        train_graphs = []
        for d in train_designs:
            train_graphs.extend(load_design_graphs(d))

        test_graphs = load_design_graphs(held_out)

        print(f"  Train: {len(train_graphs)} graphs "
              f"({', '.join(train_designs)})")
        print(f"  Test:  {len(test_graphs)} graphs ({held_out})")

        model = train_model(train_graphs)
        mae, r2, preds, actuals = evaluate(model, test_graphs)

        print(f"  MAE={mae:.4f} ns   R2={r2:.4f}")

        for pred, actual in zip(preds, actuals):
            detail_rows.append({
                "experiment":   f"LODO_{held_out}",
                "held_out":     held_out,
                "split":        "test",
                "actual_wns":   round(actual, 4),
                "predicted_wns":round(pred, 4),
                "mae_ns":       round(abs(pred - actual), 4),
            })

        summary_rows.append({
            "experiment":     f"LODO_{held_out}",
            "held_out_design": held_out,
            "train_designs":  "+".join(train_designs),
            "train_graphs":   len(train_graphs),
            "test_graphs":    len(test_graphs),
            "test_mae_ns":    round(mae, 4),
            "test_r2":        round(r2, 4) if not np.isnan(r2) else "N/A",
        })

    # ── Fixed PicoRV32 test (main paper result) ───────────────────────────────
    print(f"\n{'='*55}")
    print("Main result: train=all 5 designs, test=PicoRV32")

    train_graphs = []
    for d in all_designs:
        train_graphs.extend(load_design_graphs(d))
    train_graphs.extend(load_subgraphs("picorv32_13ns"))

    test_graphs  = [load_graph("picorv32")]
    model        = train_model(train_graphs)
    mae, r2, preds, actuals = evaluate(model, test_graphs)
    print(f"  MAE={mae:.4f} ns   R2={r2:.4f} (single point, undefined)")

    detail_rows.append({
        "experiment":    "Main_PicoRV32",
        "held_out":      "PicoRV32",
        "split":         "test",
        "actual_wns":    round(actuals[0], 4),
        "predicted_wns": round(preds[0], 4),
        "mae_ns":        round(mae, 4),
    })
    summary_rows.append({
        "experiment":      "Main_PicoRV32",
        "held_out_design": "PicoRV32",
        "train_designs":   "All 5 designs + subgraphs",
        "train_graphs":    len(train_graphs),
        "test_graphs":     1,
        "test_mae_ns":     round(mae, 4),
        "test_r2":         "N/A (single point)",
    })

    # ── LODO aggregate R2 ─────────────────────────────────────────────────────
    lodo_maes = [r["test_mae_ns"] for r in summary_rows
                 if r["experiment"].startswith("LODO")]
    lodo_r2s  = [r["test_r2"] for r in summary_rows
                 if r["experiment"].startswith("LODO")
                 and r["test_r2"] != "N/A"]

    print(f"\n{'='*55}")
    print(f"LODO summary ({len(all_designs)} experiments):")
    print(f"  Mean MAE = {np.mean(lodo_maes):.4f} ns")
    print(f"  Std  MAE = {np.std(lodo_maes):.4f} ns")
    if lodo_r2s:
        print(f"  Mean R2  = {np.mean(lodo_r2s):.4f}")
        print(f"  Std  R2  = {np.std(lodo_r2s):.4f}")
    print(f"  PicoRV32 test MAE = {summary_rows[-1]['test_mae_ns']:.4f} ns")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    with open(OUT_D / "lodo_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
        w.writeheader()
        w.writerows(detail_rows)
    print(f"\nSaved → results/data/lodo_results.csv")

    with open(OUT_D / "lodo_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved → results/data/lodo_summary.csv")

if __name__ == "__main__":
    main()
