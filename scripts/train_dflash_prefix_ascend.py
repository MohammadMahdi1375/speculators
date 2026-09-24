#!/usr/bin/env python3
"""Register the NPU backend, then use the repository's existing trainer."""

import os
import runpy
from pathlib import Path

import torch
import torch_npu  # noqa: F401

if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
