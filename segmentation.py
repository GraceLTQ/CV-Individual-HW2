import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def graph_based_segmentation(img):
    h, w, _ = img.shape
    # Create a sparse matrix for A
    A = lil_matrix((h * w, h * w))

    def index(i, j):
        # Convert 2D pixel position to 1D index
        return i * w + j

    # Fill in the weights in the adjacency matrix A
    for i in range(h):
        for j in range(w):
            for di in range(-20, 21):
                for dj in range(-20, 21):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and (di != 0 or dj != 0):
                        if abs(di) <= 20 and abs(dj) <= 20:
                            color_diff = np.linalg.norm(img[i, j] - img[ni, nj])
                            weight = np.exp(-100 * color_diff ** 2)
                            A[index(i, j), index(ni, nj)] = weight
                            A[index(ni, nj), index(i, j)] = weight

    # Convert A to a CSR format for faster arithmetic operations
    A = A.tocsr()

    # Solve the relaxed optimization problem
    # We use 'which='SM'' to find the smallest magnitude eigenvalues
    eigenvalues, eigenvectors = eigsh(A, k=2, which='SM', maxiter=1000)
    # The eigenvector corresponding to the second smallest eigenvalue
    y = eigenvectors[:, 1]
    # Reshape the resulting eigenvector back to the image shape
    segmentation = y.reshape(h, w)

    return segmentation

# Assuming `img` is a numpy array of your image loaded elsewhere
# segmentation_result = graph_based_segmentation(img)
# Now you can use `segmentation_result` for further processing or visualization
