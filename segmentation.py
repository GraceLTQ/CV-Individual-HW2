import numpy as np
from scipy.sparse import lil_matrix
from scipy.linalg import eigh  # For dense matrix eigendecomposition


def graph_based_segmentation(img, use_dense_solver=False):
    h, w, _ = img.shape
    A = lil_matrix((h * w, h * w))

    def index(i, j):
        return i * w + j

    for i in range(h):
        for j in range(w):
            for di in range(-20, 21):
                for dj in range(-20, 21):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and (di != 0 or dj != 0):
                        if abs(di) <= 20 and abs(dj) <= 20:
                            color_diff = np.linalg.norm(
                                img[i, j] - img[ni, nj])
                            weight = np.exp(-100 * color_diff ** 2)
                            A[index(i, j), index(ni, nj)] = weight
                            A[index(ni, nj), index(i, j)] = weight

    A = A.tocsr()
    # Regularization
    A += lil_matrix(np.eye(A.shape[0]) * 0.01)

    if use_dense_solver:
        # Convert to a dense matrix and use a dense solver
        A_dense = A.toarray()
        eigenvalues, eigenvectors = eigh(A_dense)
        segmentation = eigenvectors[:, 1].reshape(h, w)
    else:
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(A, k=2, which='SM', maxiter=5000)
        segmentation = eigenvectors[:, 1].reshape(h, w)

    return segmentation
