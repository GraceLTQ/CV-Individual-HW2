import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags


def graph_based_segmentation(img):
    h, w, _ = img.shape  # Height and width of the image
    N = h * w            # Total number of pixels

    # Initialize a LIL matrix since it's efficient for constructing sparse matrices
    W = np.zeros((N, N), dtype=np.float32)
    Nom = np.zeros((N, N), dtype=np.float32)
    Denom = np.zeros((N, N), dtype=np.float32)

    # Iterate through all pairs of pixels within the defined neighborhood
    for i in range(h):
        for j in range(w):
            for k in range(max(0, i-20), min(h, i+21)):
                for l in range(max(0, j-20), min(w, j+21)):
                    # Calculate the nomicator
                    # Convert the 2D coordinates to a single index for a 1D vector
                    p = i * w + j
                    q = k * w + l

                    # If p and q are the same pixel, continue to the next iteration
                    if p == q:
                        continue

                    # Calculate the weight based on the color distance
                    if abs(i-k) <= 20 and abs(j-l) <= 20:
                        color_distance = np.linalg.norm(img[i, j] - img[k, l])
                        Nom[p, q] = np.exp(-100 * color_distance ** 2)

                    # Calculate the denominator
                    Denom[p, q] = np.sum(Nom[p, :])

                    # Calculate the weight
                    W[p, q] = Nom[p, q] / Denom[p, q]

    # Construct the degree matrix D
    I = np.identity(N)

    # Construct the A matrix
    A = I - W

    # Compute the eigenvalues and eigenvectors of the A matrix
    eigenvalues, eigenvectors = eigsh(A, k=2, which='SM')

    # The second eigenvector is the eigenvector associated with the second smallest eigenvalue
    second_eigenvector = eigenvectors[:, 1]

    return second_eigenvector

# Example usage:
# img = np.array(Image.open('path_to_image.jpg'), dtype=np.float32) / 255.
# segmentation_result = graph_based_segmentation(img)
