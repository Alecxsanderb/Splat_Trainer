# Status

_Last updated: 2026-11-05_

## Next deadline
- **Nov 9**: first progress report (code + 1 page). Last chance to change scope.

## State
- Repo scaffolded with the src/ layout. Core/ext/tests files contain only the HUMAN-WRITTEN header.
- configs/dev.yaml and configs/final.yaml exist; `learning_rates` is still empty (author fills in).
- Platforms: develop on this Mac (Apple Silicon, MPS); train on a Windows 11 PC (RX 9070 XT,
  AMD PyTorch-on-Windows/ROCm). There is no Linux machine.
- Mac env: done. .venv with torch 2.15.0 (MPS available); `pip install -e .` done; check_env passes
  (float64 gradcheck works on CPU).
- Windows env: NOT set up yet. Follow README "Windows 11 + ROCm" (AMD docs are the source of truth).
  download_mipnerf360.ps1 hasn't been run or parse-checked (no PowerShell on the Mac).

## Next steps
- Set up the Windows PC and run scripts/check_env.py there (fallback: NU Explorer)
- Download data (.sh on Mac, .ps1 on Windows)
- Data loading for one scene at 8× → rasterizer forward → first seconds/iteration measurement

## Open questions
- Professor feedback on course fit and on AI-assisted Triton (none yet)
- Final numeric targets (set after the vanilla baseline)
