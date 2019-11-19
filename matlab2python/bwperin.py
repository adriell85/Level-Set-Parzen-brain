def bwperin(bw, shapes):
    import numpy as np
    rows, cols = shapes
    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))
    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (np.where(north == bw, 1, 0)) & \
          (np.where(south == bw, 1, 0)) & \
          (np.where(west == bw, 1, 0)) & \
          (np.where(east == bw, 1, 0))
    return (1 - idx) * bw
