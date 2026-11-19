# Splat Trainer

A 3D Gaussian Splatting trainer with a hand-written differentiable rasterizer in PyTorch,
plus efficiency-oriented extensions. Fall 2026 AI course final project.

## AI-use policy (summary; full rules in CLAUDE.md)
- `src/splat/` (everything except `src/splat/assisted/`) and `tests/`: **human-written only**.
  Assistants may explain concepts and point out likely bugs, but may not write code or give fixes.
- `src/splat/assisted/`, `scripts/`: AI-assisted allowed; each file starts with the header
  `# AI-ASSISTED (non-core) — see CLAUDE.md`.
- `paper/`, `reports/`: human-written prose; grammar suggestions only.

## Setup
PyTorch is **not** a dependency in `pyproject.toml`: it is installed separately for each platform.

### macOS (development)
Apple Silicon, MPS backend.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .
python scripts/check_env.py
```

### Windows 11 + ROCm (training)
RX 9070 XT, using AMD's native PyTorch-on-Windows builds. **AMD's ROCm-for-Radeon Windows
documentation is the source of truth** for versions and download locations; nothing is
pinned here because those change.

1. Manually install AMD's PyTorch-on-Windows driver package (Adrenalin edition that
   supports PyTorch), following AMD's docs.
2. Create a venv with the **specific Python version AMD's wheels require** (check the docs),
   then install PyTorch exactly as AMD's ROCm-for-Radeon Windows docs say. Do **not** use
   the pytorch.org install selector.
3. Install the package and verify:

```powershell
pip install -e .
python scripts\check_env.py
```

## Data
Mip-NeRF 360 (the release already includes `images_8`):

```bash
bash scripts/download_mipnerf360.sh            # macOS
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_mipnerf360.ps1   # Windows
```

Configs: `configs/dev.yaml` (7k iterations), `configs/final.yaml` (30k).
