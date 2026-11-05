# Splat Trainer

A 3D Gaussian Splatting trainer with a hand-written differentiable rasterizer in PyTorch,
plus efficiency-oriented extensions. Fall 2026 AI course final project.

## AI-use policy (summary; full rules in CLAUDE.md)
- `src/splat/core/`, `src/splat/ext/`, `tests/`: **human-written only**.
- `src/splat/assisted/`, `scripts/`: AI-assisted allowed; each file starts with the header
  `# AI-ASSISTED (non-core) — see CLAUDE.md`.
- `paper/`, `reports/`: human-written prose; grammar suggestions only.

## Setup (Linux + ROCm)
PyTorch is **not** a dependency in `pyproject.toml`. Install a ROCm build yourself from
the official wheel index first, then install the package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocmX.Y torch   # pick the ROCm version
pip install -e .
python scripts/check_env.py
```

Data (Mip-NeRF 360, already includes `images_8`):

```bash
bash scripts/download_mipnerf360.sh
```

Configs: `configs/dev.yaml` (7k iterations), `configs/final.yaml` (30k).
