import torch
from typing import TypeVar

input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=tuple)


# py3.8 harness shim (real task.py uses TypedDict/NotRequired from py3.11).
class TestSpec(dict):
    pass
