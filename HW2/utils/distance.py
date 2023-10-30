import numpy as np

def euclidean(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return ValueError()

    if not a.shape == b.shape:
        return ValueError()

    return np.sqrt(np.sum((a - b) ** 2))


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return ValueError()

    if not a.shape == b.shape:
        return ValueError()

    dot_prd = np.sum(a * b)
    mod_a = np.sqrt(np.sum(a ** 2))
    mod_b = np.sqrt(np.sum(b ** 2))

    return dot_prd / (mod_a * mod_b)
