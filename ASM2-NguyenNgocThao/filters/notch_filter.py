import numpy as np

def notch_filter(size, cutoff, u0, v0):
    rows, cols = size
    center = (rows // 2, cols // 2)
    filter_mask = np.ones((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            D1 = np.sqrt((u - center[0] - u0) ** 2 + (v - center[1] - v0) ** 2)
            D2 = np.sqrt((u - center[0] + u0) ** 2 + (v - center[1] + v0) ** 2)
            if D1 <= cutoff or D2 <= cutoff:
                filter_mask[u, v] = 0
    return filter_mask
