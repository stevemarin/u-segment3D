import numpy as np
import scipy.ndimage as ndimage
from . import CALL_LOG


def _summ(v):
    if isinstance(v, np.ndarray):
        return ("array", v.shape, str(v.dtype), float(np.nansum(v)))
    if isinstance(v, (list, tuple)):
        return tuple(_summ(x) for x in v)
    return v


def _eval(imgs, **kwargs):
    CALL_LOG.append(
        tuple(sorted((k, _summ(v)) for k, v in kwargs.items())) + (("img", _summ(imgs[0])),)
    )
    img = np.asarray(imgs[0], dtype=np.float64)
    if img.ndim == 3:
        img = img.max(axis=-1)
    diam = kwargs.get("diameter")
    d = 17.0 if diam is None else float(diam)
    prob = ndimage.gaussian_filter(img, d / 10.0) * 10 - 3
    dP = np.array(np.gradient(ndimage.gaussian_filter(img, d / 8.0)))
    style = np.linspace(0, 1, 16) * img.mean()
    flows = [[None, dP.astype(np.float32), prob.astype(np.float32)]]
    if diam is None:
        return None, flows, [style], np.array(d)
    return None, flows, [style]


class _CP:
    eval = staticmethod(_eval)


class Cellpose:
    def __init__(self, model_type=None, gpu=False):
        self.cp = _CP()


class CellposeModel:
    def __init__(self, pretrained_model=None, gpu=False):
        pass

    eval = staticmethod(_eval)
