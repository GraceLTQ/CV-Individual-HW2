import numpy as np


def graph_based_segmentation(img):
    h, w, _ = np.shape(img)  # Height and width of the image
    N = h * w            # Total number of pixels

    W = np.zeros((N, N))  # Weight matrix
    y1, x1 = np.unravel_index(np.arange(h*w), (h, w))
    y2, x2 = np.unravel_index(np.arange(h*w), (h, w))

    for r in range(N):
        py, px = y1[r], x1[r]
        for c in range(N):
            qy, qx = y2[c], x2[c]

            if (px == qx and py == qy):
                W[r, c] = 0
            elif (np.abs(px - qx) <= 20 and np.abs(py - qy) <= 20):
                color_p = img[py, px, :]
                color_q = img[qy, qx, :]
                W[r][c] = np.exp(-100 * (np.linalg.norm(color_p - color_q))**2)

    denom = np.sum(W, axis=1)
    I = np.identity(N)

    for i in range(N):
        W[i, :] = W[i, :] / denom[i]

    A = I - W

    eigenvalues, eigenvectors = np.linalg.eig(A)
    second_eigenvalue = (np.argsort(eigenvalues))[1]
    second_eigenvector = eigenvectors[:, second_eigenvalue]

    return np.reshape(second_eigenvector, (h, w))
