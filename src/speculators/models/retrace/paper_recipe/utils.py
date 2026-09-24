import os
import signal
import subprocess
import torch
import torch.distributed as dist

def average_gradients(parameters, count, device, distributed):
    total = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(total)
    if total.item() == 0:
        return 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        if distributed:
            dist.all_reduce(gradient)
        gradient.div_(total)
        parameter.grad = gradient
    return int(total.item())

def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

