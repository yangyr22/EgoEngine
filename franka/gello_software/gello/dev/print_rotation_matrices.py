import itertools
import numpy as np


def generate_transformation_matrices():
    base_permutations = list(itertools.permutations([1, 0, 0])) + \
                        list(itertools.permutations([0, 1, 0])) + \
                        list(itertools.permutations([0, 0, 1]))

    # Filter to only take permutations with exactly 3 elements
    base_permutations = [list(p) for p in set(itertools.permutations(base_permutations, 3))]

    matrices = []
    for perm in base_permutations:
        for signs in itertools.product([1, -1], repeat=3):
            signed_perm = [np.array(row) * sign for row, sign in zip(perm, signs)]
            matrix = np.array(signed_perm)
            if (np.abs(matrix).sum(axis=0) == 1).all() and (np.abs(matrix).sum(axis=1) == 1).all():
                matrices.append(matrix)

    return matrices


def print_matrices(matrices):
    for matrix in matrices:
        print("NEXT =================================")
        print(matrix)
        print()


if __name__ == "__main__":
    matrices = generate_transformation_matrices()
    print_matrices(matrices)
