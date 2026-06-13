"""
Benchmark: per-line vs batched VietOCR recognition, on CPU vs GPU.

Runs the full DeepDoc+VietOCR OCR pipeline over benchmark/images and reports
per-page and total wall-clock for each configuration.

Usage:
    python benchmark/run_benchmark.py                 # all 4 configs (GPU skipped if no CUDA)
    python benchmark/run_benchmark.py --device cpu    # CPU only
    python benchmark/run_benchmark.py --images path/to/dir

The recognition path is controlled by two env vars read inside module/ocr.py:
    VIETOCR_DEVICE  = cpu | cuda:0       (recognizer device)
    VIETOCR_BATCH   = 1 (batched) | 0 (one line at a time)
    CUDA_VISIBLE_DEVICES                 (set to "" to force CPU, "0" to expose GPU)
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_config(images, device, batch):
    """Run t_ocr.py over `images` in a fresh process with the given config."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["VIETOCR_DEVICE"] = device
    env["VIETOCR_BATCH"] = "1" if batch else "0"
    env["CUDA_VISIBLE_DEVICES"] = "0" if device.startswith("cuda") else ""
    out_dir = os.path.join(HERE, "out_%s_%s" % (device.replace(":", ""), "batch" if batch else "perline"))
    cmd = [sys.executable, os.path.join(REPO, "t_ocr.py"),
           "--inputs", images, "--output_dir", out_dir]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    return elapsed, ok, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(HERE, "images"))
    ap.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    args = ap.parse_args()

    n = len(glob.glob(os.path.join(args.images, "*.*")))
    configs = []
    want_gpu = args.device in ("cuda", "both") and have_cuda()
    want_cpu = args.device in ("cpu", "both")
    if want_cpu:
        configs += [("cpu", False), ("cpu", True)]
    if want_gpu:
        configs += [("cuda:0", False), ("cuda:0", True)]
    if args.device in ("cuda", "both") and not have_cuda():
        print("(CUDA not available — skipping GPU configs)")

    print("Benchmarking %d images from %s\n" % (n, args.images))
    rows = []
    for device, batch in configs:
        label = "%s / %s" % (device, "batched" if batch else "per-line")
        print("Running: %s ..." % label, flush=True)
        elapsed, ok, _ = run_config(args.images, device, batch)
        status = "" if ok else "  (FAILED)"
        rows.append((label, elapsed, n, ok))
        print("  -> %.1f s total, %.2f s/page%s\n" % (elapsed, elapsed / max(n, 1), status))

    print("\n## Results (%d pages)\n" % n)
    print("| Config | Total (s) | Avg s/page |")
    print("|---|---|---|")
    for label, elapsed, count, ok in rows:
        print("| %s | %.1f | %.2f |" % (label, elapsed, elapsed / max(count, 1)))


if __name__ == "__main__":
    main()
