"""
====================================================================
 RNA Secondary Structure Prediction using Dynamic Programming (Nussinov-style)
====================================================================

This program predicts the secondary structure of a given RNA sequence by finding
the maximum number of non-crossing base pairs, following the constraints:

 - Valid base pairs: A-U, C-G, and wobble G-U
 - No sharp turns: pairs must be at least 4 bases apart (j - i ≥ 4)
 - Each base can pair with at most one other base
 - No pseudoknots: base pairs cannot cross (non-crossing condition)

The algorithm uses dynamic programming to compute an optimal pairing matrix and
generates the resulting structure in **dot-bracket notation**.

"""
def is_valid_pair(base1: str, base2: str) -> bool:
    """
    Check if two RNA bases can form a valid pair.
    Supports Watson-Crick (A-U, C-G) and wobble (G-U) pairings.
    """
    return (base1 == 'A' and base2 == 'U') or (base1 == 'U' and base2 == 'A') or \
           (base1 == 'C' and base2 == 'G') or (base1 == 'G' and base2 == 'C') or \
           (base1 == 'G' and base2 == 'U') or (base1 == 'U' and base2 == 'G')


def print_opt_table(opt_val):
    """
    Utility function to print the dynamic programming table (opt_val)
    in a readable, lower-triangular format.
    """
    n = len(opt_val)
    for i in range(n - 6, -1, -1):
        print(f"{i:2}:", end=' ')
        for j in range(5, n):
            print(str(opt_val[i][j]).rjust(2), end=' ')
        print()
    print(" j:", end=' ')
    for j in range(5, n):
        print(str(j).rjust(2), end=' ')
    print('\n')


def get_opt_val(seq, n, opt_val, opt_val_pairs, i, j):
    """
    Recursively compute the optimal number of non-crossing base pairs
    between positions i and j in the RNA sequence using dynamic programming.
    Also tracks the optimal base pairs in opt_val_pairs.
    """
    # No valid pair if distance is too small (sharp turn constraint)
    if j - i < 4:
        opt_val[i][j] = 0
        return 0

    # Return precomputed result
    if opt_val[i][j] != -1:
        return opt_val[i][j]

    # Case 1: j is unpaired
    max_val = get_opt_val(seq, n, opt_val, opt_val_pairs, i, j - 1)
    best_pairs = opt_val_pairs[i][j - 1]

    # Case 2: try pairing j with every t in [i, j-5] if valid
    for t in range(i, j - 4):
        if is_valid_pair(seq[t], seq[j]):
            left = get_opt_val(seq, n, opt_val, opt_val_pairs, i, t - 1) if t - 1 >= i else 0
            right = get_opt_val(seq, n, opt_val, opt_val_pairs, t + 1, j - 1) if t + 1 <= j - 1 else 0
            val = 1 + left + right
            if val > max_val:
                max_val = val
                left_pairs = opt_val_pairs[i][t - 1] if t - 1 >= i else []
                right_pairs = opt_val_pairs[t + 1][j - 1] if t + 1 <= j - 1 else []
                best_pairs = left_pairs + [(t, j)] + right_pairs

    # Store and return result
    opt_val[i][j] = max_val
    opt_val_pairs[i][j] = best_pairs
    return max_val


def get_dot_bracket_notation(n, pairs):
    """
    Generate dot-bracket notation from the base pair indices.
    """
    notation = ['.'] * n
    for i, j in pairs:
        notation[i] = '('
        notation[j] = ')'
    return ''.join(notation)


def compute_rna_secondary_structure(sequence, debug=False):
    """
    Main function that computes the RNA secondary structure
    using dynamic programming and returns:
    - dot-bracket notation
    - base pair indices
    - base pair values
    - DP table
    """
    n = len(sequence)

    # Initialize DP tables
    opt_val = [[-1 for _ in range(n)] for _ in range(n)]
    opt_val_pairs = [[[] for _ in range(n)] for _ in range(n)]

    # Base cases: no valid structure for substrings shorter than 5
    for i in range(n):
        for j in range(n):
            if j - i < 4:
                opt_val[i][j] = 0

    # Fill DP table from bottom-up
    for k in range(5, n):
        for i in range(0, n - k):
            j = i + k
            get_opt_val(sequence, n, opt_val, opt_val_pairs, i, j)

    # Get final optimal structure for full sequence
    get_opt_val(sequence, n, opt_val, opt_val_pairs, 0, n - 1)
    final_pairs = opt_val_pairs[0][n - 1]
    dot_bracket = get_dot_bracket_notation(n, final_pairs)
    pair_bases = [(sequence[i], sequence[j]) for (i, j) in final_pairs]

    # Optional debug output
    if debug:
        print_opt_table(opt_val)
        print("Base Pairs (i, j):", final_pairs)
        print("Base Pairs (bases):", pair_bases)
        print("Dot-Bracket Notation:", dot_bracket)

    return dot_bracket, final_pairs, pair_bases, opt_val


# === RUNNING EXAMPLE ===
if __name__ == "__main__":
    # Example RNA sequence
    rna_seq = "ACAUGAUGGCCAUGU"
    notation, pairs, base_pairs, opt_table = compute_rna_secondary_structure(rna_seq, debug=True)

    print("\nFinal Result:")
    print("RNA Sequence:        ", rna_seq)
    print("Dot-Bracket Notation:", notation)
