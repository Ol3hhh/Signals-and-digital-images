import cupy as cp
import numpy as np


def quantization(pixels, Q):
    # return cp.transpose(cp.floor(Q*pixels + 0.5) / Q)
    return cp.transpose(cp.floor(Q * pixels + 0.5 - (cp.sign(pixels) / 4)) / Q)