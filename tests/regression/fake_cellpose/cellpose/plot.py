import numpy as np


def dx_to_circ(dP):
    return np.zeros(dP.shape[1:] + (3,), dtype=np.uint8)
