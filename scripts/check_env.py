# AI-ASSISTED (non-core) — see CLAUDE.md
"""Report the Python/PyTorch/GPU environment and run a small GPU smoke test.

Intended target: Linux + PyTorch ROCm build on an RX 9070 XT (gfx1201).
No workarounds are attempted on other platforms; problems are just reported.
"""

import platform
import sys
import time


def main() -> int:
    os_name = platform.system()
    print(f"OS:            {os_name} {platform.release()} ({platform.machine()})")
    print(f"Python:        {sys.version.split()[0]} ({sys.executable})")
    if os_name != "Linux":
        print(f"WARNING: OS is {os_name}, not Linux. ROCm PyTorch is Linux-only; "
              "this machine is not the intended training target.")

    try:
        import torch
    except ImportError:
        print("torch:         NOT INSTALLED")
        return 1

    hip = getattr(torch.version, "hip", None)
    print(f"torch:         {torch.__version__}")
    print(f"torch.version.hip: {hip if hip else 'not set (not a ROCm build)'}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        print("No GPU visible to torch; skipping GPU test.")
        return 1

    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"Device:        {props.name}")
    arch = getattr(props, "gcnArchName", None)
    print(f"gfx arch:      {arch if arch else 'unavailable'}")

    n = 2048
    a = torch.randn(n, n, device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn(n, n, device=dev, dtype=torch.float32, requires_grad=True)

    # Warm-up (kernel selection / lazy init).
    (a @ b).sum().backward()
    torch.cuda.synchronize()
    a.grad = b.grad = None

    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    torch.cuda.synchronize()
    fwd = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        (a @ b).sum().backward()
    torch.cuda.synchronize()
    fwd_bwd = (time.perf_counter() - t0) / reps

    tflops = 2 * n**3 / fwd / 1e12
    print(f"fp32 matmul {n}x{n}: fwd {fwd * 1e3:.2f} ms (~{tflops:.1f} TFLOP/s), "
          f"fwd+bwd {fwd_bwd * 1e3:.2f} ms")
    grads_ok = a.grad is not None and torch.isfinite(a.grad).all().item()
    print(f"Backward grads finite: {grads_ok}")
    del c
    return 0 if grads_ok else 1


if __name__ == "__main__":
    sys.exit(main())
