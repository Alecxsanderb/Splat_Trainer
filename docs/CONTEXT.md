# Context: Custom 3DGS Trainer (AI course final project)

## Summary
Fall 2026 AI course final project (individual). A 3D Gaussian Splatting trainer with a
hand-written differentiable rasterizer in PyTorch, plus efficiency-oriented extensions.
Long-term motivation: replacing Lustre's dependence on Luma AI with an in-house trainer,
eventually on-device. Proposal accepted without feedback.

## Deadlines
- Oct 26: Literature review (1–2 pages, at least one peer-reviewed paper)
- Nov 9: First progress report (code + 1 page). Last chance to change scope.
- Nov 23: Final progress report (code must run without errors + brief writeup)
- Dec 7: Final submission (code + paper, weighted equally; paper ~5–10 pages double-spaced)

## AI policy (binding, applies to any assistant working on this repo)
- Core functionality must be human-written: rasterizer, trainer, loss, densification,
  data loading/preprocessing, evaluation, and all extension logic. Assistants may explain
  concepts and review, but must not write this code.
- Non-core code (training-video tooling, Lustre/Swift integration) may be AI-assisted,
  kept in separate files and clearly labeled.
- No AI writing of the paper or checkpoint reports. Grammar edits only, with the
  original text in an appendix.
- An AI-ported Triton kernel is NOT approved. Ask the professor before considering it.

## Architecture
- Rasterizer (PyTorch, autograd backward):
  - project Gaussians, with 2D covariance Σ' = J W Σ Wᵀ Jᵀ and Σ = R S Sᵀ Rᵀ
  - bin into 16×16 tiles, depth-sort
  - alpha-composite front to back
  - must be fully vectorized, with no Python loops over tiles
- Trainer:
  - loss: (1−λ)·L1 + λ·D-SSIM, with λ = 0.2
  - Adam with per-parameter-group learning rates; position LR decays
  - SH degree ramps up every 1000 iterations
  - adaptive density control (clone, split, prune, opacity reset) in the first half of training
- Shared infrastructure for the extensions:
  - per-Gaussian importance accumulator (Σ α·T over views)
  - per-view, per-tile error cache

## Extensions (tiered)
1. Committed: importance-based view sampling and visibility pruning.
2. Committed if the core lands on schedule: error-driven tile skipping, with periodic full
   passes over a view-diverse subset (farthest-point sampling over cameras). Per-Gaussian
   staleness invalidation is an experiment against that baseline.
3. Stretch: LOD hierarchy (merge small Gaussians for distance) from the importance and
   footprint statistics.

Optional performance tier: `torch.compile` first. A hand-written Triton compositing kernel
(forward + backward, wrapped in an autograd.Function) only if seconds/iteration demands
it and time allows.

## Data & compute
- Mip-NeRF 360 (provided COLMAP poses). Every 8th image held out for test.
  Likely 8× downsampled. Start with a few scenes, at least one indoor and one outdoor.
- Lustre captures: future work, since the capture system isn't built yet.
  ARKit/LiDAR initialization experiments depend on it.
- Home GPU: RX 9070 XT with PyTorch on ROCm (verify; ROCm is Linux-first).
  Fallback: NU Explorer cluster, which is also needed for gsplat comparisons (CUDA-only).
- First key measurement: seconds/iteration at 8× on one scene.
  Development runs at 7k iterations, final runs at 30k.

## Evaluation
- Correctness:
  - forward renders match gsplat (~40+ dB PSNR between the two renders)
  - `gradcheck` passes on tiny scenes
- Quality: PSNR, SSIM, and LPIPS on held-out views, compared to gsplat or Brush under
  matched resolution and iteration count
- Efficiency: iterations-to-target-PSNR, Gaussian count, and fraction of views/tiles
  processed, all compared to the vanilla trainer
- Ablations: initialization (random vs. SfM), each extension on vs. off
- Training video: periodic parameter snapshots rendered from a fixed orbit
  (tooling is non-core)

## References
- Anchor: Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field
  Rendering," ACM TOG (SIGGRAPH)
- Candidates (verify venues before citing):
  - Taming 3DGS (SIGGRAPH Asia 2024)
  - LightGaussian (NeurIPS 2024)
  - Hierarchical 3DGS (SIGGRAPH 2024)
  - Revising Densification in Gaussian Splatting (ECCV 2024)
  - 3DGS-MCMC (NeurIPS 2024)
- Course tie-in: Local Search and Optimization slides (objective functions, hill climbing,
  local maxima, random restarts); Russell & Norvig, local search in continuous spaces

## Open questions
- Course-fit and AI-assisted Triton questions: no professor feedback received yet
- Final numeric targets: set after the vanilla baseline runs
