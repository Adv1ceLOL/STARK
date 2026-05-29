from permuted_tree import merkelize, mk_branch, verify_branch, mk_multi_branch, verify_multi_branch
from utils import get_power_cycle, get_pseudorandom_indices
from poly_utils import PrimeField

# Generate an FRI proof that the polynomial that has the specified
# values at successive powers of the specified root of unity has a
# degree lower than maxdeg_plus_1
#
# We use maxdeg+1 instead of maxdeg because it's more mathematically
# convenient in this case.

def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p

def _merkelize_padded(values):
    """Merkelize values, padding to next power of 2 with zero-bytes if needed.
    Returns (tree, padded_size)."""
    n = len(values)
    padded_n = _next_power_of_2(n)
    if padded_n != n:
        # Pad with zero values for Merkle tree compatibility
        if isinstance(values[0], int):
            padded = list(values) + [0] * (padded_n - n)
        else:
            padded = list(values) + [b'\x00' * 4] * (padded_n - n)
        return merkelize(padded), padded_n
    return merkelize(values), n


def _fold_by_127(values, xs, f, modulus, exclude_multiples_of=0):
    """
    Perform a single 127-fold step in FRI.
    
    This reduces a polynomial evaluated on a domain of size 127*m to a
    polynomial evaluated on a domain of size m, by interpolating through
    groups of 127 points and evaluating at a random challenge.
    
    Args:
        values: List of field elements, length = 127 * m
        xs: Power cycle (domain points) of length 127 * m
        f: PrimeField instance
        modulus: Prime modulus
        exclude_multiples_of: Optional exclusion parameter
        
    Returns:
        (column, special_x, m_tree, m_padded_size, ys, poly_positions):
        The folded column and the proof data for this step.
    """
    n = len(values)
    assert n % 127 == 0
    m = n // 127  # size after folding (power of 2)
    
    # Merkelize current values
    m_tree, m_padded_size = _merkelize_padded(values)
    
    # Derive random challenge from Merkle root
    special_x = int.from_bytes(m_tree[1], 'big') % modulus
    
    # Fold by 127: for each position j in [0, m), collect the 127 values
    # at positions j, j+m, j+2m, ..., j+126*m. These are evaluations of
    # the original polynomial at xs[j], xs[j+m], xs[j+2m], ..., xs[j+126*m].
    # Interpolate through these 127 points and evaluate at special_x.
    
    column = []
    
    # NEW OPTIMIZATION: Barycentric Weight Scaling
    # The x-coordinates for any row j are scalar multiples of the x-coordinates for row 0.
    # Because barycentric evaluation normalizes the weights by their sum, scalar multipliers
    # on the weights cancel out exactly. This means we can compute the weights ONCE
    # for row 0 and reuse them for all m iterations!
    # This eliminates O(127^2) field operations *per iteration*.
    base_x_points = [xs[k * m] for k in range(127)]
    shared_weights = f.precompute_barycentric_weights(base_x_points)
    
    for j in range(m):
        x_points = [xs[j + k * m] for k in range(127)]
        y_points = [values[j + k * m] for k in range(127)]
        
        # Barycentric evaluation using shared weights: O(127) operations total!
        column.append(f.barycentric_eval_at(x_points, y_points, special_x, shared_weights))
    
    # Merkelize the folded column
    m2_tree, m2_padded_size = _merkelize_padded(column)
    
    # Select random sample positions for verification
    ys = get_pseudorandom_indices(m2_tree[1], len(column), 40,
                                  exclude_multiples_of=exclude_multiples_of)
    
    # For each sample position y in the folded column, the verifier needs
    # the 127 original values at positions y, y+m, y+2m, ..., y+126*m
    poly_positions = sum([[y + m * k for k in range(127)] for y in ys], [])
    
    return column, special_x, m_tree, m2_tree, ys, poly_positions, m


def prove_low_degree(values, root_of_unity, maxdeg_plus_1, modulus, exclude_multiples_of=0):
    f = PrimeField(modulus)
    print('Proving %d values are degree <= %d' % (len(values), maxdeg_plus_1))

    # If the degree we are checking for is less than or equal to 32,
    # use the polynomial directly as a proof
    # Base case: small degree OR domain not divisible by 4 (can't quartic-fold)
    if maxdeg_plus_1 <= 16:
        print('Produced FRI proof')
        return [[x.to_bytes(4, 'big') for x in values]]

    # === NEW: If domain size is divisible by 127 but not a power of 2, ===
    # === fold by 127 first to get to a pure power-of-2 domain         ===
    if len(values) % 127 == 0 and (len(values) & (len(values) - 1)) != 0:
        n = len(values)
        m = n // 127  # This should be a power of 2
        
        # Calculate x coordinates
        xs = get_power_cycle(root_of_unity, modulus)
        assert len(values) == len(xs), f"Power cycle size {len(xs)} != values size {len(values)}"
        
        # Perform 127-fold
        column, special_x, m_tree, m2_tree, ys, poly_positions, m_out = \
            _fold_by_127(values, xs, f, modulus, exclude_multiples_of)
        
        # Build proof component for the 127-fold step
        # Format: [fold_factor, m2_root, column_branches, poly_branches]
        o = [127, m2_tree[1], mk_multi_branch(m2_tree, ys), mk_multi_branch(m_tree, poly_positions)]
        
        # The new root of unity is omega^127 (advances by 127, reducing domain by 127×)
        new_root_of_unity = f.exp(root_of_unity, 127)
        new_maxdeg = maxdeg_plus_1 // 127
        if new_maxdeg < 1:
            new_maxdeg = 1
        
        # Recurse on the folded column (now pure power-of-2 domain!)
        return [o] + prove_low_degree(column, new_root_of_unity, new_maxdeg, modulus,
                                       exclude_multiples_of=exclude_multiples_of)
    
    # === Standard quartic folding (power-of-2 domain) ===
    if len(values) % 4 != 0:
        print('Produced FRI proof')
        return [[x.to_bytes(4, 'big') for x in values]]

    # Calculate the set of x coordinates
    xs = get_power_cycle(root_of_unity, modulus)
    assert len(values) == len(xs)

    # Put the values into a Merkle tree. This is the root that the
    # proof will be checked against
    m, m_padded_size = _merkelize_padded(values)

    # Select a pseudo-random x coordinate
    special_x = int.from_bytes(m[1], 'big') % modulus

    # Calculate the "column" at that x coordinate
    # (see https://vitalik.ca/general/2017/11/22/starks_part_2.html)
    # We calculate the column by Lagrange-interpolating each row, and not
    # directly from the polynomial, as this is more efficient
    quarter_len = len(xs)//4
    x_polys = f.multi_interp_4(
        [[xs[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)],
        [[values[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)]
    )
    column = [f.eval_quartic(p, special_x) for p in x_polys]
    m2, m2_padded_size = _merkelize_padded(column)

    # Pseudo-randomly select y indices to sample
    ys = get_pseudorandom_indices(m2[1], len(column), 40, exclude_multiples_of=exclude_multiples_of)

    # Compute the positions for the values in the polynomial
    poly_positions = sum([[y + (len(xs) // 4) * j for j in range(4)] for y in ys], [])

    # This component of the proof, including Merkle branches
    o = [m2[1], mk_multi_branch(m2, ys), mk_multi_branch(m, poly_positions)]

    # Recurse...
    return [o] + prove_low_degree(column, f.exp(root_of_unity, 4),
                                  maxdeg_plus_1 // 4, modulus, exclude_multiples_of=exclude_multiples_of)

# Verify an FRI proof
def verify_low_degree_proof(merkle_root, root_of_unity, proof, maxdeg_plus_1, modulus, exclude_multiples_of=0):
    f = PrimeField(modulus)

    # Calculate the order of the root of unity efficiently
    # FIXED: Instead of O(n) loop, use the known structure of the domain
    # For mixed-radix domains (127 * 2^k), compute order from factorization
    roudeg = _compute_root_order(root_of_unity, modulus)
    
    # Verify the recursive components of the proof
    for prf in proof[:-1]:
        # Detect whether this is a 127-fold step or a quartic fold step
        if isinstance(prf[0], int) and prf[0] == 127:
            # === 127-fold verification ===
            fold_factor, root2, column_branches, poly_branches = prf
            print('Verifying degree <= %d (127-fold)' % maxdeg_plus_1)
            
            m = roudeg // 127  # domain size after folding
            
            # Calculate the pseudo-random x coordinate
            # First we need to reconstruct the Merkle root of the current layer
            # For the 127-fold, the Merkle root is derived from the values
            # We need to verify Merkle branches against the current merkle_root
            special_x = int.from_bytes(merkle_root, 'big') % modulus
            
            # Calculate sample positions
            ys = get_pseudorandom_indices(root2, m, 40,
                                          exclude_multiples_of=exclude_multiples_of)
            poly_positions = sum([[y + m * k for k in range(127)] for y in ys], [])
            
            # Verify Merkle branches
            column_values = verify_multi_branch(root2, ys, column_branches)
            poly_values = verify_multi_branch(merkle_root, poly_positions, poly_branches)
            
            # Precompute shared barycentric weights ONCE for all verifier checks
            base_x_points = [f.exp(root_of_unity, k * m) for k in range(127)]
            shared_weights = f.precompute_barycentric_weights(base_x_points)
            m_root = base_x_points[1] if len(base_x_points) > 1 else 1
            
            # For each sampled position, verify the 127-fold interpolation
            for i, y in enumerate(ys):
                # Get the 127 x-coordinates using optimized exponentiation
                base_x = f.exp(root_of_unity, y)
                x_points = [0] * 127
                curr_x = base_x
                for k in range(127):
                    x_points[k] = curr_x
                    curr_x = (curr_x * m_root) % modulus
                
                # Get the 127 y-values from the proof
                row = [int.from_bytes(x, 'big') for x in poly_values[i*127: i*127+127]]
                
                # Barycentric evaluation using shared weights: O(127) ops
                expected_val = f.barycentric_eval_at(x_points, row, special_x, shared_weights)
                
                actual_val = int.from_bytes(column_values[i], 'big')
                assert expected_val == actual_val, \
                    f"127-fold verification failed at position {y}: expected {expected_val}, got {actual_val}"
            
            # Update for next round
            merkle_root = root2
            root_of_unity = f.exp(root_of_unity, 127)
            maxdeg_plus_1 = maxdeg_plus_1 // 127
            if maxdeg_plus_1 < 1:
                maxdeg_plus_1 = 1
            roudeg //= 127
            
        else:
            # === Standard quartic fold verification ===
            root2, column_branches, poly_branches = prf
            print('Verifying degree <= %d' % maxdeg_plus_1)

            # Powers of the given root of unity 1, p, p**2, p**3 such that p**4 = 1
            quartic_roots_of_unity = [1,
                                      f.exp(root_of_unity, roudeg // 4),
                                      f.exp(root_of_unity, roudeg // 2),
                                      f.exp(root_of_unity, roudeg * 3 // 4)]

            # Calculate the pseudo-random x coordinate
            special_x = int.from_bytes(merkle_root, 'big') % modulus

            # Calculate the pseudo-randomly sampled y indices
            ys = get_pseudorandom_indices(root2, roudeg // 4, 40,
                                          exclude_multiples_of=exclude_multiples_of)

            # Compute the positions for the values in the polynomial
            poly_positions = sum([[y + (roudeg // 4) * j for j in range(4)] for y in ys], [])

            # Verify Merkle branches
            column_values = verify_multi_branch(root2, ys, column_branches)
            poly_values = verify_multi_branch(merkle_root, poly_positions, poly_branches)

            # For each y coordinate, get the x coordinates on the row, the values on
            # the row, and the value at that y from the column
            xcoords = []
            rows = []
            columnvals = []
            for i, y in enumerate(ys):
                # The x coordinates from the polynomial
                x1 = f.exp(root_of_unity, y)
                xcoords.append([(quartic_roots_of_unity[j] * x1) % modulus for j in range(4)])

                # The values from the original polynomial
                row = [int.from_bytes(x, 'big') for x in poly_values[i*4: i*4+4]]
                rows.append(row)

                columnvals.append(int.from_bytes(column_values[i], 'big'))

            # Verify for each selected y coordinate that the four points from the
            # polynomial and the one point from the column that are on that y 
            # coordinate are on the same deg < 4 polynomial
            polys = f.multi_interp_4(xcoords, rows)

            for p, c in zip(polys, columnvals):
                assert f.eval_quartic(p, special_x) == c

            # Update constants to check the next proof
            merkle_root = root2
            root_of_unity = f.exp(root_of_unity, 4)
            maxdeg_plus_1 //= 4
            roudeg //= 4

    # Verify the direct components of the proof
    data = [int.from_bytes(x, 'big') for x in proof[-1]]
    print('Verifying degree <= %d' % maxdeg_plus_1)
    assert maxdeg_plus_1 <= 16, \
        f"Invalid FRI base case: maxdeg_plus_1={maxdeg_plus_1}, len(data)={len(data)}"

    # Check the Merkle root matches up
    mtree, _ = _merkelize_padded(data)
    assert mtree[1] == merkle_root

    # Check the degree of the data
    powers = get_power_cycle(root_of_unity, modulus)
    if exclude_multiples_of:
        pts = [x for x in range(len(data)) if x % exclude_multiples_of]
    else:
        pts = range(len(data))

    poly = f.lagrange_interp([powers[x] for x in pts[:maxdeg_plus_1]],
                             [data[x] for x in pts[:maxdeg_plus_1]])
    for x in pts[maxdeg_plus_1:]:
        assert f.eval_poly_at(poly, powers[x]) == data[x]   

    print('FRI proof verified')
    return True


def _compute_root_order(root_of_unity, modulus):
    """
    Compute the multiplicative order of root_of_unity mod modulus.
    
    OPTIMIZED: Instead of the O(n) iterative loop, uses the known structure
    of the Koala prime (p-1 = 2^24 * 127) to compute the order efficiently.
    
    The order must divide p-1 = 2^24 * 127. So we check:
    - What is the largest k such that root^(2^24 * 127 / 2^k) != 1? → 2-adic part
    - Does root^(2^24) == 1? If not, 127 divides the order
    
    This runs in O(24 + 1) = O(25) exponentiations instead of O(n) multiplications.
    """
    p_minus_1 = modulus - 1  # 2^24 * 127
    
    # Check if 127 divides the order: root^(p-1/127) = root^(2^24) should be != 1
    val_without_127 = pow(root_of_unity, p_minus_1 // 127, modulus)
    has_127 = (val_without_127 != 1)
    
    # Find the 2-adic part: root^(p-1/2^k) for decreasing k
    # The 2-adic valuation of the order
    two_adic = 0
    for k in range(24, -1, -1):
        exp = p_minus_1 // (1 << k)
        if has_127:
            pass  # handled below
        if pow(root_of_unity, exp, modulus) != 1:
            two_adic = k + 1
            break
    
    # The order is 2^two_adic * (127 if has_127 else 1)
    # But we need to verify more carefully for mixed cases
    # Actually, find exact order by trial division of p-1
    
    order = p_minus_1
    # Try dividing out factors of 2
    while order % 2 == 0:
        if pow(root_of_unity, order // 2, modulus) == 1:
            order //= 2
        else:
            break
    # Try dividing out 127
    if order % 127 == 0:
        if pow(root_of_unity, order // 127, modulus) == 1:
            order //= 127
    
    return order
