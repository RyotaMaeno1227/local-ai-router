#!/usr/bin/env python3
"""Check local Python, PyTorch, CUDA, GPU, and package availability."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from typing import Iterable


PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_packages(packages: Iterable[str]) -> None:
    print("\nPython packages:")
    for package in packages:
        print(f"  {package}: {package_version(package)}")


def main() -> None:
    print("Runtime:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print(f"  Platform: {platform.platform()}")

    print_packages(PACKAGES)

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - environment diagnostic
        print(f"\nPyTorch import failed: {exc}")
        raise SystemExit(1) from exc

    print("\nPyTorch/CUDA:")
    print(f"  torch.__version__: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  cudnn version: {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device is visible to PyTorch.")
        print("Check WSL GPU visibility with `nvidia-smi` outside this script.")
        return

    print(f"  device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / (1024**3)
        print(f"\nCUDA device {idx}:")
        print(f"  name: {props.name}")
        print(f"  capability: {props.major}.{props.minor}")
        print(f"  total memory: {total_gib:.2f} GiB")

    try:
        print(f"\nBF16 supported: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:  # pragma: no cover - environment diagnostic
        print(f"\nBF16 support check failed: {exc}")


if __name__ == "__main__":
    main()
