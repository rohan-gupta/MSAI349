def euclidean(a, b):
    if len(a) != len(b):
        return 0

    dist = 0

    for i, val in enumerate(a):
        dist += (a[i] - b[i]) ** 2

    return dist ** 0.5


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    if len(a) != len(b):
        return 0

    dot_prd = 0
    mod_a = 0
    mod_b = 0

    for i, val in enumerate(a):
        dot_prd += a[i] * b[i]
        mod_a += a[i] ** 2
        mod_b += b[i] ** 2

    mod_a = mod_a ** 0.5
    mod_b = mod_b ** 0.5

    return dot_prd / (mod_a * mod_b)
