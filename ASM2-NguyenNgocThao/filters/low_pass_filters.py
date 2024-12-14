import numpy as np

def ideal_lowpass_filter(size, cutoff):
    rows, cols = size
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            if D <= cutoff:
                filter_mask[u, v] = 1
    return filter_mask

def gaussian_lowpass_filter(size, cutoff):
    rows, cols = size
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            filter_mask[u, v] = np.exp(-(D ** 2) / (2 * (cutoff ** 2)))
    return filter_mask

def butterworth_lowpass_filter(size, cutoff, order=2):
    rows, cols = size
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            filter_mask[u, v] = 1 / (1 + (D / cutoff) ** (2 * order))
    return filter_mask
