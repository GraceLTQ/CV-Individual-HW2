import numpy as np


def graph_based_segmentation(img):
    h, w, _ = np.shape(img)  # Height and width of the image
    N = h * w            # Total number of pixels

    # Initialize a LIL matrix since it's efficient for constructing sparse matrices
    W = np.zeros((N, N))
    y1, x1 = np.unravel_index(np.arange(h*w), (h, w))
    y2, x2 = np.unravel_index(np.arange(h*w), (h, w))

    # Iterate through all pairs of pixels within the defined neighborhood
    for i in range(N):
        py = y1[i]
        px = x1[i]
        for j in range(N):
            qy = y2[j]
            qx = x2[j]
            if (py == qy and px == qx):
                W[i, j] = 0
            elif (np.abs(px - qx) <= 20) and (np.abs(py - qy) <= 20):
                colorp = img[py, px, :]
                colorq = img[qy, qx, :]
                W[i][j] = np.exp(-100 * (np.linalg.norm(colorp - colorq))**2)

    Denom = np.sum(W, axis=1)

    # Construct the degree matrix D
    I = np.identity(N)

    for i in range(N):
        # if Denom[i] != 0:
        W[i, :] = W[i, :] / Denom[i]

    # Construct the A matrix
    A = I - W

    # Compute the eigenvalues and eigenvectors of the A matrix
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # The second eigenvector is the eigenvector associated with the second smallest eigenvalue
    second_eigenvalue = (np.argsort(eigenvalues))[1]

    second_eigenvector = eigenvectors[:, second_eigenvalue]

    # Reshape the second eigenvector to match the original image dimensions before returning
    return np.reshape(second_eigenvector, (h, w))


# Example usage:
# img = np.array(Image.open('path_to_image.jpg'), dtype=np.float32) / 255.
# segmentation_result = graph_based_segmentation(img)
