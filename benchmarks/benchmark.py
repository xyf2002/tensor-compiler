#!/usr/bin/env python3
"""
benchmark.py - CPU (fused vs unfused) vs GPU benchmark for TensorOps matmul+relu.

Usage:
    python3 benchmarks/benchmark.py [--sizes 64 128 256 512] [--iterations 5]

CPU pipeline (two-stage lowering: tensor-opt for custom passes,
mlir-opt for standard MLIR passes):
  unfused: tensor-opt --tensor-to-linalg
  fused:   tensor-opt --fuse-matmul-relu --tensor-to-linalg
  both:    → mlir-opt (bufferize, linalg→loops→cf→llvm)
           → mlir-translate → lli -O0

GPU pipeline:
  tensor-opt --tensor-to-linalg --lower-tensor-ops-to-gpu-nvvm
  → extract gpu.module → mlir-translate → llc → ptxas → gpu-runner
"""

import subprocess
import sys
import os
import csv
import time
import argparse
import tempfile
import re
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TENSOR_OPT = os.environ.get(
    "TENSOR_OPT",
    str(PROJECT_ROOT / "build" / "tools" / "tensor-opt" / "tensor-opt"),
)
TENSOR_RUNNER = os.environ.get(
    "TENSOR_RUNNER",
    str(PROJECT_ROOT / "build" / "tools" / "tensor-runner" / "tensor-runner"),
)
GPU_RUNNER = os.environ.get(
    "GPU_RUNNER",
    str(PROJECT_ROOT / "build" / "tools" / "gpu-runner" / "gpu-runner"),
)
MLIR_OPT = os.environ.get("MLIR_OPT", "/usr/lib/llvm-20/bin/mlir-opt")
MLIR_TRANSLATE = os.environ.get("MLIR_TRANSLATE", "mlir-translate")
LLC = os.environ.get("LLC", "llc")
LLI = os.environ.get("LLI", "/usr/lib/llvm-20/bin/lli")
PTXAS = os.environ.get("PTXAS", "ptxas")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV",
                            str(PROJECT_ROOT / "benchmarks" / "results.csv"))

SEED = 42


def make_matmul_relu_mlir(size: int) -> str:
    return (
        f"func.func @matmul_relu("
        f"%a: tensor<{size}x{size}xf32>, "
        f"%b: tensor<{size}x{size}xf32>) "
        f"-> tensor<{size}x{size}xf32> {{\n"
        f"  %0 = ten.matmul %a, %b "
        f": tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32> "
        f"-> tensor<{size}x{size}xf32>\n"
        f"  %1 = ten.relu %0 "
        f": tensor<{size}x{size}xf32> -> tensor<{size}x{size}xf32>\n"
        f"  func.return %1 : tensor<{size}x{size}xf32>\n"
        f"}}\n"
    )


def make_gpu_matmul_relu_mlir(size: int) -> str:
    return (
        f"func.func @matmul("
        f"%a: tensor<{size}x{size}xf32>, "
        f"%b: tensor<{size}x{size}xf32>) "
        f"-> tensor<{size}x{size}xf32> {{\n"
        f"  %0 = ten.matmul %a, %b "
        f": tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32> "
        f"-> tensor<{size}x{size}xf32>\n"
        f"  %1 = ten.relu %0 "
        f": tensor<{size}x{size}xf32> -> tensor<{size}x{size}xf32>\n"
        f"  func.return %1 : tensor<{size}x{size}xf32>\n"
        f"}}\n"
    )


def build_wrapper_mlir(size: int, target_func: str) -> str:
    """Build @main wrapper with embedded random data and side-effecting store
    to prevent DCE (mirrors tensor-runner's approach)."""
    rng = random.Random(SEED)
    a_data = [[rng.uniform(-1.0, 1.0) for _ in range(size)] for _ in range(size)]
    b_data = [[rng.uniform(-1.0, 1.0) for _ in range(size)] for _ in range(size)]

    def fmt_tensor(data):
        rows = []
        for row in data:
            vals = ", ".join(f"{v:.8e}" for v in row)
            rows.append(f"[{vals}]")
        return "[" + ",\n    ".join(rows) + "]"

    lines = []
    lines.append("func.func @main() -> i32 {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append(
        f"  %a = arith.constant dense<{fmt_tensor(a_data)}>"
        f" : tensor<{size}x{size}xf32>"
    )
    lines.append(
        f"  %b = arith.constant dense<{fmt_tensor(b_data)}>"
        f" : tensor<{size}x{size}xf32>"
    )
    lines.append(
        f"  %c = call @{target_func}(%a, %b)"
        f" : (tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32>)"
        f" -> tensor<{size}x{size}xf32>"
    )
    lines.append(
        f"  %m = bufferization.to_memref %c"
        f" : tensor<{size}x{size}xf32> to memref<{size}x{size}xf32>"
    )
    lines.append(
        f"  %v = memref.load %m[%c0, %c0]"
        f" : memref<{size}x{size}xf32>")
    lines.append(
        f"  memref.store %v, %m[%c0, %c0]"
        f" : memref<{size}x{size}xf32>")
    lines.append("  %ret = arith.constant 0 : i32")
    lines.append("  return %ret : i32")
    lines.append("}")
    return "\n".join(lines)


def make_full_cpu_mlir(size: int) -> str:
    wrapper = build_wrapper_mlir(size, "matmul_relu")
    func = make_matmul_relu_mlir(size)
    return wrapper + "\n\n" + func


_MLIR_OPT_PASSES = [
    "--one-shot-bufferize=bufferize-function-boundaries=true",
    "--buffer-results-to-out-params",
    "--empty-tensor-to-alloc-tensor",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--finalize-memref-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]


def run_cpu(mlir_str: str, size: int, fuse: bool = True,
            iterations: int = 10, warmup: int = 2) -> float:
    """CPU benchmark: tensor-opt -> mlir-opt -> mlir-translate -> lli.
    Returns avg ms per iteration, or float('inf') on failure."""
    full_mlir = make_full_cpu_mlir(size)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir",
                                     delete=False) as f:
        f.write(full_mlir)
        mlir_path = f.name

    step1_path = mlir_path + ".step1.mlir"
    step2_path = mlir_path + ".step2.mlir"
    ll_path = mlir_path + ".ll"

    try:
        # Step 1: tensor-opt -- only custom TensorOps passes (tensor-to-linalg,
        # optionally fuse-matmul-relu). Standard MLIR passes are NOT registered
        # in tensor-opt, so they go through mlir-opt in Step 2.
        opt_cmd = [TENSOR_OPT]
        if fuse:
            opt_cmd.append("--fuse-matmul-relu")
        opt_cmd.extend(["--tensor-to-linalg", mlir_path, "-o", step1_path])

        r = subprocess.run(opt_cmd, capture_output=True, text=True,
                           timeout=120)
        if r.returncode != 0:
            _dbg("tensor-opt failed", r.stderr, fuse, size)
            return float("inf")

        # Step 2: mlir-opt -- standard lowering pipeline (matching
        # tensor-runner's buildLoweringPipeline).
        r = subprocess.run(
            [MLIR_OPT] + _MLIR_OPT_PASSES + [step1_path, "-o", step2_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            _dbg("mlir-opt failed", r.stderr, fuse, size)
            return float("inf")

        # Step 3: mlir-translate -> LLVM IR
        r = subprocess.run(
            [MLIR_TRANSLATE, "--mlir-to-llvmir", step2_path, "-o", ll_path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            _dbg("mlir-translate failed", r.stderr, fuse, size)
            return float("inf")

        # Step 4: lli JIT execution
        run_cmd = [LLI, "-O0", ll_path]

        for _ in range(warmup):
            r = subprocess.run(run_cmd, capture_output=True, timeout=300)
            if r.returncode != 0:
                _dbg("lli warmup failed", r.stderr, fuse, size)
                return float("inf")

        elapsed = 0.0
        for i in range(iterations):
            t0 = time.perf_counter()
            r = subprocess.run(run_cmd, capture_output=True, timeout=300)
            t1 = time.perf_counter()
            if r.returncode != 0:
                _dbg(f"lli iteration {i} failed", r.stderr, fuse, size)
                return float("inf")
            elapsed += t1 - t0

        avg_ms = (elapsed / iterations) * 1000.0
        return avg_ms

    except subprocess.TimeoutExpired:
        _dbg("timed out", "", fuse, size)
        return float("inf")
    except FileNotFoundError as e:
        _dbg(f"binary not found: {e}", "", fuse, size)
        return float("inf")
    finally:
        for p in [mlir_path, step1_path, step2_path, ll_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def run_gpu(mlir_str: str, size: int,
            iterations: int = 10) -> float:
    """GPU benchmark: tensor-opt -> extract gpu.module -> mlir-translate
    -> llc -> ptxas -> gpu-runner. Returns ms from CUDA events or inf."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir",
                                     delete=False) as f:
        f.write(mlir_str)
        mlir_path = f.name

    nvvm_path = mlir_path + ".nvvm"
    gpu_mlir_path = nvvm_path + ".gpu.mlir"
    ll_path = mlir_path + ".ll"
    ptx_path = mlir_path + ".ptx"
    cubin_path = mlir_path + ".cubin"

    try:
        r = subprocess.run(
            [TENSOR_OPT, "--tensor-to-linalg",
             "--lower-tensor-ops-to-gpu-nvvm",
             mlir_path, "-o", nvvm_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            _dbg("tensor-opt (gpu) failed", r.stderr, False, size)
            return float("inf")

        rc = _extract_gpu_module(nvvm_path, gpu_mlir_path)
        if rc != 0:
            _dbg("gpu module extraction failed", "", False, size)
            return float("inf")

        r = subprocess.run(
            [MLIR_TRANSLATE, "--mlir-to-llvmir", gpu_mlir_path,
             "-o", ll_path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            _dbg("mlir-translate (gpu) failed", r.stderr, False, size)
            return float("inf")

        r = subprocess.run(
            [LLC, "-march=nvptx64", "-mcpu=sm_89",
             ll_path, "-o", ptx_path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            _dbg("llc failed", r.stderr, False, size)
            return float("inf")

        r = subprocess.run(
            [PTXAS, "-arch=sm_89", ptx_path, "-o", cubin_path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            _dbg("ptxas failed", r.stderr, False, size)
            return float("inf")

        kernel_name = "matmul_kernel"
        r = subprocess.run(
            [GPU_RUNNER, cubin_path, kernel_name, str(size)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            _dbg("gpu-runner failed", r.stderr, False, size)
            return float("inf")

        for line in r.stdout.split("\n"):
            m = re.search(r"Kernel execution time.*?([\d.]+)\s*ms", line)
            if m:
                return float(m.group(1))
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                return float(m.group(1))

        _dbg("could not parse gpu timing", r.stdout, False, size)
        return float("inf")

    except subprocess.TimeoutExpired:
        _dbg("gpu pipeline timed out", "", False, size)
        return float("inf")
    except FileNotFoundError as e:
        _dbg(f"gpu binary not found: {e}", "", False, size)
        return float("inf")
    finally:
        for p in [mlir_path, nvvm_path, gpu_mlir_path,
                  ll_path, ptx_path, cubin_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def _extract_gpu_module(nvvm_path: str, output_path: str) -> int:
    script = f"""
import sys, re
with open('{nvvm_path}') as f:
    content = f.read()
parts = content.split('gpu.module @')
if len(parts) < 2:
    print('Error: No gpu.module found', file=sys.stderr)
    sys.exit(1)
gpu_part = parts[1]
depth = 0
end = 0
for i, c in enumerate(gpu_part):
    if c == '{{':
        depth += 1
    elif c == '}}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
gpu_module = (
    'module attributes {{gpu.container_module}} {{\\n'
    '  gpu.module @kernel {{\\n'
    + gpu_part[:end] +
    '\\n  }}\\n'
    '}}'
)
with open('{output_path}', 'w') as f:
    f.write(gpu_module)
print('OK')
"""
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return r.returncode
    return 0 if "OK" in r.stdout else 1




def _dbg(msg: str, detail: str, fuse: bool, size: int):
    label = "fused" if fuse else "unfused"
    short = detail.strip()[:120] if detail else ""
    print(f"  [WARN] CPU {label} size={size}: {msg} {short}",
          file=sys.stderr)


def fmt_val(v: float, suffix: str = "") -> str:
    if v == float("inf") or v == 0.0:
        return f"{'FAILED':>10}"
    return f"{v:>10.3f}{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorOps matmul+relu"
    )
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[64, 128, 256, 512],
                        help="Matrix sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of timed iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU benchmarks")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="Output CSV file path")
    args = parser.parse_args()

    sizes = args.sizes
    iterations = args.iterations
    warmup = args.warmup

    print("=" * 70)
    print("  TensorOps Benchmark  —  matmul+relu fused vs unfused vs GPU")
    print("=" * 70)
    print(f"  Matrix sizes : {sizes}")
    print(f"  Iterations   : {iterations}  (warmup: {warmup})")
    print()

    results = []

    for size in sizes:
        print(f"  [{size}x{size}]")

        print("    CPU unfused ... ", end="", flush=True)
        t0 = time.perf_counter()
        unfused = run_cpu(None, size, fuse=False,
                          iterations=iterations, warmup=warmup)
        wall = time.perf_counter() - t0
        if unfused != float("inf"):
            print(f"{unfused:.3f} ms  (wall: {wall:.1f}s)")
        else:
            print("FAILED")

        print("    CPU fused   ... ", end="", flush=True)
        t0 = time.perf_counter()
        fused = run_cpu(None, size, fuse=True,
                        iterations=iterations, warmup=warmup)
        wall = time.perf_counter() - t0
        if fused != float("inf"):
            print(f"{fused:.3f} ms  (wall: {wall:.1f}s)")
        else:
            print("FAILED")

        gpu_time = float("inf")
        if not args.no_gpu:
            print("    GPU         ... ", end="", flush=True)
            gpu_mlir = make_gpu_matmul_relu_mlir(size)
            t0 = time.perf_counter()
            gpu_time = run_gpu(gpu_mlir, size,
                               iterations=iterations)
            wall = time.perf_counter() - t0
            if gpu_time != float("inf"):
                print(f"{gpu_time:.3f} ms  (wall: {wall:.1f}s)")
            else:
                print("FAILED")

        # Compute speedups
        fusion_sp = (
            unfused / fused
            if fused > 0 and fused != float("inf") and unfused != float("inf")
            else float("inf")
        )
        gpu_vs_cpu = (
            unfused / gpu_time
            if gpu_time > 0 and gpu_time != float("inf")
               and unfused != float("inf")
            else float("inf")
        )

        results.append({
            "size": size,
            "cpu_unfused_ms": unfused,
            "cpu_fused_ms": fused,
            "gpu_ms": gpu_time,
            "fusion_speedup": fusion_sp,
            "gpu_vs_cpu_unfused": gpu_vs_cpu,
        })
        print()

    print()
    print("-" * 80)
    print(
        f"{'Size':>6} | {'CPU Unfused':>14} | {'CPU Fused':>12} | "
        f"{'GPU':>10} | {'Fusion Sp':>10} | {'GPU/CPU':>8}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['size']:>6} | "
            f"{fmt_val(r['cpu_unfused_ms'], 'ms'):>14} | "
            f"{fmt_val(r['cpu_fused_ms'], 'ms'):>12} | "
            f"{fmt_val(r['gpu_ms'], 'ms'):>10} | "
            f"{fmt_val(r['fusion_speedup'], 'x'):>10} | "
            f"{fmt_val(r['gpu_vs_cpu_unfused'], 'x'):>8}"
        )
    print("-" * 80)

    fieldnames = [
        "size", "cpu_unfused_ms", "cpu_fused_ms", "gpu_ms",
        "fusion_speedup", "gpu_vs_cpu_unfused",
    ]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n  Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
