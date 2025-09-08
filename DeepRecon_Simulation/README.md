# DeepRecon (VGG16) — Flush+Reload Simulation

This file contains a self‑contained **PyTorch** notebook that simulates a **Flush+Reload‑style side‑channel attack** to **recover the high‑level architecture of VGG16**. It builds a canonical VGG16 model, logs a *ground‑truth* operator sequence via forward hooks, **simulates noisy cache‑probe traces**, **denoises** them, and finally **parses** the events with a small **finite‑state machine (FSM)** tailored to VGG to reconstruct the network. It then compares **Recovered vs Ground‑Truth** attributes.

> Notebook: `deeprecon.ipynb`

---

## What it does

1. **Build VGG16(+Softmax) & log ops** — A lightweight wrapper (`VGG16WithSoftmax`) around `torchvision.models.vgg16` adds a terminal Softmax and installs forward hooks to record the **ground‑truth op sequence**.
2. **Simulate Flush+Reload probes** — `simulate_trace(...)` generates a **noisy probe trace** over framework‑level ops with configurable timing, hit/miss latencies, drop/false‑positive rates, etc.
3. **Denoise the trace** — `denoise(...)` performs a **per‑function debounce** and filters by a latency threshold to keep plausible events.
4. **FSM parsing for VGG** — `vgg_fsm_parse(...)` consumes denoised events and enforces a **VGG grammar** (Conv→[BiasAdd]?→ReLU→…→MaxPool; FC stages; final Softmax).
5. **Report metrics** — `count_attrs(...)` summarizes **8 attributes** (convs, fcs, relus, pools, softmaxes, biases, …), block structure (**convs per block**), and shows **GT vs Recovered** (with an L1 error helper).

---

## 📁 Files

- `deeprecon.ipynb` — The full simulation, utilities, and analysis.

---

## Requirements

- Python 3.9+
- PyTorch and TorchVision
- NumPy

Install (CPU‑only example):

```bash
pip install numpy torch torchvision
```

For CUDA builds, see the official PyTorch installation selector.

---

## How to run

1. **Clone your repo** and open the notebook:
   ```bash
   git clone <your-repo-url>.git
   cd <your-repo>
   jupyter notebook deeprecon.ipynb
   ```
2. **Run top‑to‑bottom**. The notebook:
   - Builds VGG16 and logs a GT op sequence via hooks.
   - Simulates a probe trace with noise.
   - Denoises and FSM‑parses events.
   - Prints a **Recovered vs GT** comparison table and block structure (expected VGG16: `[2, 2, 3, 3, 3]` convs per block).
3. **Tweak noise/latencies** in `simulate_trace(...)` to stress the pipeline.

---

## How it works (high level)

- **Ground truth logging:** Forward hooks map modules to coarse op names (`Conv2D`, `ReLU`, `MaxPool`, `FC`, `BiasAdd`, `Softmax`).  
- **Probe simulation:** A scheduled list of op windows is probed at a fixed interval, emitting **hit/miss latencies** with user‑set probabilities and jitter.  
- **Denoising:** Converts latencies → hits, applies **per‑function debounce** and plausibility filtering.  
- **FSM parsing:** A tiny **VGG FSM** parses the event stream to a **valid architecture sequence**, discarding out‑of‑grammar noise.  
- **Comparison:** Attribute counts and block structure are compared against GT; a small helper reports **L1 errors** per attribute.

---

## Expected output

- A printed table with **GT vs Recovered** for the 8 attributes.  
- **Convs per block** vector close to **`[2, 2, 3, 3, 3]`**.  
- Example artifacts: first 30 GT ops, latency histogram (hits vs misses), event counts before/after denoising.

> Stochastic components (seeds, noise) may cause small run‑to‑run differences; set `set_seed(7)` for reproducibility.

---

## Key functions & classes

- `VGG16WithSoftmax(nn.Module)` — wraps `torchvision.models.vgg16` and appends Softmax.  
- `record_ops(model, x)` — registers hooks and returns the **GT op sequence**.  
- `simulate_trace(gt_seq, durations_us, ...)` — generates probe records with **hit/miss latencies**.  
- `denoise(records, ...)` — thresholds latencies + per‑function debounce → chronological events.  
- `vgg_fsm_parse(events, convs_per_block)` — parses events under the **VGG grammar**.  
- `count_attrs(seq)` — returns the 8‑attribute summary.  
- `main()` — end‑to‑end run and report.
