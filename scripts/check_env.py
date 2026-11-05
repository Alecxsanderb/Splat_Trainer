# AI-ASSISTED (non-core) — see CLAUDE.md
"""Report the Python/PyTorch/GPU environment and run small smoke tests.

Targets:
  - macOS (Apple Silicon): development, MPS backend
  - Windows 11 + RX 9070 XT: training, AMD's PyTorch-on-Windows (ROCm) build
  - Linux: reported the same way as Windows

A missing GPU backend is a warning, not an error.
"""

import platform
import sys
import time


def bench(torch, device, n: int = 1024, reps: int = 10) -> str:
    """Time an fp32 n×n matmul forward and forward+backward on `device`."""

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    a = torch.randn(n, n, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(n, n, device=device, dtype=torch.float32, requires_grad=True)

    # Warm-up (lazy init / kernel selection).
    (a @ b).sum().backward()
    sync()
    a.grad = b.grad = None

    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    sync()
    fwd = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        (a @ b).sum().backward()
    sync()
    fwd_bwd = (time.perf_counter() - t0) / reps

    finite = bool(torch.isfinite(a.grad).all().item())
    tflops = 2 * n**3 / fwd / 1e12
    return (f"fp32 matmul {n}x{n}: fwd {fwd * 1e3:.2f} ms (~{tflops:.2f} TFLOP/s), "
            f"fwd+bwd {fwd_bwd * 1e3:.2f} ms, grads finite: {finite}")


def report_gpu(torch, os_name: str):
    """Print backend info; return a torch.device for the GPU, or None."""
    if os_name == "Darwin":
        built = torch.backends.mps.is_built()
        avail = torch.backends.mps.is_available()
        print(f"MPS built:     {built}")
        print(f"MPS available: {avail}")
        return torch.device("mps") if avail else None

    hip = getattr(torch.version, "hip", None)
    print(f"torch.version.hip:         {hip if hip else 'not set (not a ROCm build)'}")
    avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {avail}")
    if not avail:
        return None
    props = torch.cuda.get_device_properties(0)
    print(f"Device:        {props.name}")
    arch = getattr(props, "gcnArchName", None)
    print(f"gfx arch:      {arch if arch else 'n/a'}")
    return torch.device("cuda", 0)


def main() -> int:
    os_name = platform.system()
    print(f"OS:            {os_name} {platform.release()} ({platform.machine()})")
    print(f"Python:        {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed in this environment.")
        return 1
    print(f"torch:         {torch.__version__}")

    print("\n== GPU backend ==")
    gpu = report_gpu(torch, os_name)

    print("\n== Benchmarks ==")
    if gpu is not None:
        print(f"[{gpu.type}] {bench(torch, gpu)}")
    else:
        print("WARNING: no GPU backend available; skipping GPU benchmark.")
    print(f"[cpu] {bench(torch, torch.device('cpu'))}")

    print("\n== float64 on CPU (for gradcheck) ==")
    x = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda t: (t @ t).sin(), (x,))
    print(f"float64 tensor dtype: {x.dtype}, gradcheck on toy fn: {ok}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
