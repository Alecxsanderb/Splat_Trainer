# Status

_Last updated: 2026-11-04_

## Next deadline
- **Nov 9**: first progress report (code + 1 page). Last chance to change scope.

## State
- Repo scaffolded with the src/ layout. Core/ext/tests files contain only the HUMAN-WRITTEN header.
- configs/dev.yaml and configs/final.yaml exist; `learning_rates` is still empty (author fills in).
- Env: NOT set up. The scaffold was created on macOS (no ROCm), so check_env reports no GPU.
  On the Linux box: create .venv, install a ROCm torch wheel, `pip install -e .`, run scripts/check_env.py.

## Next steps
- Verify PyTorch on ROCm with the RX 9070 XT (fallback: NU Explorer)
- `bash scripts/download_mipnerf360.sh`
- Data loading for one scene at 8× → rasterizer forward → first seconds/iteration measurement

## Open questions
- Professor feedback on course fit and on AI-assisted Triton (none yet)
- Final numeric targets (set after the vanilla baseline)
