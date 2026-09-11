# Shim, written by me, for the upstream GPU MODE `task.py`.
# Upstream uses TypedDict/NotRequired (py3.11); this is a py3.8 stand-in with
# the same names. Covered by this repo's MIT license. See ../NOTICE.
import torch
from typing import TypeVar

input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=tuple)


# py3.8 harness shim (real task.py uses TypedDict/NotRequired from py3.11).
class TestSpec(dict):
    pass
