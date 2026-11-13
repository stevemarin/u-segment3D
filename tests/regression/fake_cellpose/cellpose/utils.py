import numpy as np
import scipy.ndimage as ndimage


def fill_holes_and_remove_small_masks(masks, min_size=15):
    masks = masks.copy()
    for lab in np.setdiff1d(np.unique(masks), 0):
        m = masks == lab
        if m.sum() < min_size:
            masks[m] = 0
        else:
            masks[ndimage.binary_fill_holes(m)] = lab
    return masks
