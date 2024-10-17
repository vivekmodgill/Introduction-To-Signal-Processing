def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        # Make a copy of the vector to avoid modifying the original data
        v_copy = np.copy(v)
        for b in basis:
            print('b=', b)
            v_copy -= np.dot(v_copy, b) / np.dot(b, b) * b
        # Normalize the vector to ensure it has unit length
        v_copy /= np.linalg.norm(v_copy)
        basis.append(v_copy)
    return np.array(basis)
