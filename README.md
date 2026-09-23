# FlowMind

**Pre-Route Timing Prediction via GraphSAGE on Post-Synthesis Netlists**

> ISPD 2027 · In Preparation &nbsp;|&nbsp; Open-source · Reproducible · No commercial tools required

---

## Key Results

| Metric | Value |
|---|---|
| GNN test MAE (PicoRV32, unseen) | **0.27 ns** |
| XGBoost baseline MAE (same test) | 2.90 ns |
| Improvement over baseline | **10.8×** |
| GNN inference time | < 50 ms |
| OpenSTA full run (same design) | ~45 seconds |
| Runtime reduction | **~900×** |
| Training graphs | 118 (18 full + 100 subgraphs) |
| Designs | GCD · UART · SPI · AES-ORFS · AES · PicoRV32 |
| Technology | SkyWater 130nm HD |

---

## What is FlowMind?

FlowMind predicts **worst negative slack (WNS)** before routing, using only the post-synthesis gate-level netlist. No placement coordinates, no wire length estimates, no parasitic information — just the netlist graph.

A synthesized netlist is a directed graph: cells are nodes, net connections are edges. FlowMind trains a **GraphSAGE** model on this graph to predict whether a design will meet timing, at a fraction of the cost of running STA.

The core claim: **graph topology alone encodes enough structural information to generalize timing prediction across unseen designs.** The GNN achieves 0.27 ns MAE on PicoRV32, a RISC-V processor it has never seen, while XGBoost — which relies on design identity as a feature — fails at 2.90 ns MAE on the same test.

---

## Pipeline

```
RTL Source
    │
    ▼
Yosys synth -flatten          →  flat JSON netlist
    │                              (cells + port directions + net IDs)
    ▼
Graph Construction            →  PyG Data object
    │  nodes = cell instances       5-dim node features:
    │  edges = net connections       [cell_type_enc, is_seq,
    │  label = WNS from OpenSTA      fanout_norm, fanin_norm,
    │                                pin_count_norm]
    ▼
GraphSAGE (3 layers, 64-dim)
    │  mean aggregation
    │  global mean + max pool
    ▼
MLP head  →  predicted WNS (ns)
```

OpenSTA provides ground-truth WNS labels **during training only**. At inference, the GNN runs in < 50 ms with no STA required.

---

## Repository Structure

```
flowmind/
│
├── run_synthesis.py          # Extension 1: RTL → synthesis → STA → QoR DB
├── run_pnr.py                # Extension 2: P&R via OpenROAD
├── run_signoff.py            # Extension 2b: MCMM STA signoff (SS/TT/FF)
├── run_qual.py               # Extension 3: tool qualification suite
├── run_log_triage.py         # Extension 4a: TF-IDF log search engine
├── run_slack_predictor.py    # Extension 4b: XGBoost WNS predictor
├── run_gnn_prep.py           # Extension 5: netlist JSON → PyG graphs
├── run_gnn_train.py          # Extension 5: GraphSAGE training
├── run_gnn_eval.py           # Extension 5: GNN vs XGBoost eval report
├── run_gnn_subgraph.py       # Extension 5b: BFS subgraph sampling
├── run_gnn_lodo.py           # Extension 5c: leave-one-design-out eval
│
├── configs/                  # Per-design YAML (clock, I/O delays, PDK paths)
├── templates/                # Jinja2 templates (Yosys, SDC, OpenSTA)
├── scripts/                  # QoR parser, DB writer, dashboard generator
│
├── ext5-gnn/
│   ├── netlists/             # Flat Yosys JSON + Sky130-mapped GL Verilog
│   └── graphs/               # Serialized PyG Data objects (.pt)
│                             # 19 full graphs + 100 subgraphs
│
├── results/
│   ├── data/                 # All result CSVs (reproduced from models)
│   ├── figures/              # Publication figures (PNG, 300 DPI)
│   └── scripts/              # Scripts to regenerate all CSVs and figures
│
├── models/
│   ├── slack_predictor.pkl   # Trained XGBoost model
│   └── gnn_slack.pt          # Trained GraphSAGE checkpoint
│
├── outputs/                  # Per-design STA reports and DRC reports
└── results/qor_runs.db       # SQLite: 35 synthesis + STA runs
```

---

## Extensions

### Extension 1 — Synthesis Flow Automation

Single-command RTL-to-STA with Jinja2-templated Yosys + OpenSTA scripts, QoR parsing, SQLite tracking, and Plotly dashboard.

```bash
python run_synthesis.py --design gcd
python run_synthesis.py --design picorv32 --period 13.0
python run_synthesis.py --design uart --period 5.0
```

| Design | Cells | Area (µm²) | Critical Path (ns) | Max Freq |
|---|---|---|---|---|
| PicoRV32 | 8,041 | 77,250 | 12.36 | 76 MHz |
| AES-128 | 997 | 18,401 | 27.21 | 36 MHz |
| GCD | 27 | 68 | 3.91 | 212 MHz |
| UART | 343 | 3,705 | 4.75 | 166 MHz |
| SPI | 57 | 910 | 1.35 | 481 MHz |

### Extension 2 — Place & Route + MCMM Signoff

OpenROAD P&R via Docker. MCMM signoff across SS/TT/FF corners with OCV derating.

```bash
python run_pnr.py --design gcd
python run_signoff.py --design gcd
```

**GCD result:** WNS = 0 ns · Area = 2,431 µm² · Utilization = 42% · **0 DRC violations**

### Extension 3 — Tool Qualification Suite

```bash
python run_qual.py
```

6/6 checks passed on GCD and PicoRV32 across medium/high synthesis effort.

### Extension 4a — ML Log Triage

TF-IDF search engine over 36 EDA log files (5,749 chunks). Sub-second query.

```bash
python run_log_triage.py --reindex --query "WNS violation timing slack"
python run_log_triage.py --query "DRC error metal spacing" --top 5
```

### Extension 4b — XGBoost Slack Predictor

```bash
python run_slack_predictor.py
```

Trained on 22 unique (design, clock) pairs. PicoRV32 excluded from training.
CV MAE = 0.56 ± 0.10 ns · R² = 0.93

### Extension 5 — GNN Pre-Route Timing Prediction

```bash
# Build graphs from netlists
python run_gnn_prep.py

# Generate subgraph augmentations (BFS sampling)
python run_gnn_subgraph.py

# Train GraphSAGE
python run_gnn_train.py

# Evaluate vs XGBoost baseline
python run_gnn_eval.py
```

---

## Results

### Main Result: GNN vs XGBoost on Unseen Design

| Model | Train MAE | **Test MAE (PicoRV32)** |
|---|---|---|
| XGBoost (feature-based) | 1.47 ns | 2.90 ns |
| **GraphSAGE GNN** | **1.64 ns** | **0.27 ns** |

PicoRV32 is held out from all training. Neither model sees it during training or hyperparameter tuning.

### Ablation: Training Set Size vs Test MAE

| Stage | Graphs | Test MAE (ns) | Reduction |
|---|---|---|---|
| GCD only | 1 | 10.12 | — |
| + 3 designs | 4 | 2.54 | −75% |
| + clock variants | 18 | 0.96 | −62% |
| + subgraph sampling | 118 | **0.27** | **−72%** |

### Dataset Summary

| Design | Nodes | Edges | Seq (%) | WNS Range (ns) | Split |
|---|---|---|---|---|---|
| GCD | 303 | 551 | 0 | −0.90 to +13.59 | Train |
| UART | 562 | 1,153 | 57 | −1.14 to +0.43 | Train |
| SPI | 58 | 109 | 64 | −0.41 to +0.28 | Train |
| AES-ORFS | 10,469 | 27,184 | 44 | −1.60 to +0.45 | Train |
| AES | 24,689 | 49,465 | 44 | −0.03 | Train |
| Subgraphs (×100) | 100–500 | varies | — | −1.60 to +0.45 | Train |
| **PicoRV32** | **8,038** | **18,360** | **58** | **−0.16** | **Test** |

### Reproducing Results

All result files live in `results/` and can be regenerated from scratch:

```bash
# Regenerate all CSVs from trained models and graph files
python results/scripts/generate_data.py

# Regenerate all figures
python results/scripts/fig2_wns_distribution.py
python results/scripts/fig3_loss_curve.py
python results/scripts/fig4_ablation.py
```

| File | Contents |
|---|---|
| `results/data/dataset.csv` | All 119 graphs — nodes, edges, WNS, split |
| `results/data/eval_results.csv` | Per-design GNN vs XGBoost MAE |
| `results/data/ablation.csv` | 4-stage training set ablation |
| `results/data/training_history.csv` | MSE loss per epoch (300 epochs) |
| `results/data/lodo_results.csv` | Leave-one-design-out detail |
| `results/data/lodo_summary.csv` | LODO summary per experiment |

---

## Quick Start

```bash
# Clone
git clone https://github.com/das-subrata/flowmind
cd flowmind

# Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run synthesis on any design
python run_synthesis.py --design gcd

# Build GNN dataset and train
python run_gnn_prep.py
python run_gnn_subgraph.py
python run_gnn_train.py
python run_gnn_eval.py
```

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Yosys | 0.52 | `sudo apt install yosys` |
| OpenSTA | 2.0.17 | `sudo apt install opensta` |
| OpenROAD | latest | Docker: `openroad/flow-ubuntu22.04-builder` |
| SkyWater 130nm PDK | sky130hd | Via OpenROAD-flow-scripts |
| Python | 3.10+ | — |
| PyTorch | 2.x CPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| PyTorch Geometric | latest | `pip install torch-geometric` |

Tested on Ubuntu 22.04 (WSL2). No GPU required.

---

## Tool Stack

| Tool | Role | Commercial Equivalent |
|---|---|---|
| Yosys 0.52 | RTL synthesis, JSON netlist export | Cadence Genus |
| OpenSTA 2.0.17 | Static timing analysis, WNS labels | Cadence Tempus / Synopsys PrimeTime |
| OpenROAD | Place and route | Cadence Innovus |
| SkyWater 130nm | Standard cell library | TSMC / Samsung PDK |
| PyTorch Geometric | GraphSAGE implementation | — |
| XGBoost | Feature-based baseline predictor | — |

---

## Paper

> **FlowMind: Pre-Route Timing Prediction via GraphSAGE on Post-Synthesis Netlists Using Open-Source EDA Tools**
> Subrata Das · ISPD 2027 (in preparation)

```bibtex
@misc{flowmind2026,
  title   = {FlowMind: Pre-Route Timing Prediction via GraphSAGE
             on Post-Synthesis Netlists Using Open-Source EDA Tools},
  author  = {Das, Subrata},
  year    = {2026},
  url     = {https://github.com/das-subrata/flowmind}
}
```

---

## Commit History

```
5d5d9a7  Extension 5b: subgraph sampling - 100 subgraphs, GNN 0.27ns vs XGBoost 2.90ns
04ab34d  Extension 5:  fair eval - GNN 0.96ns vs XGBoost 2.90ns (picorv32 held out)
9947385  Extension 4:  ML log triage (TF-IDF) + XGBoost slack predictor
c268119  Extension 3:  Tool qualification suite - 6/6 checks passed
e4be568  Extension 2:  MCMM STA signoff across SS/TT/FF corners
9f42127  Extension 1:  OpenROAD P&R flow - GCD, RTL to routed layout
```

---

*Built with Yosys · OpenSTA · OpenROAD · SkyWater 130nm · PyTorch Geometric*
