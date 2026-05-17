
from permuted_tree import merkelize, mk_branch, verify_branch, mk_multi_branch, verify_multi_branch
from utils import get_power_cycle as orig_get_power_cycle, get_pseudorandom_indices
from poly_utils import PrimeField, ExtensionField

def serialize(x):
    if isinstance(x, tuple):
        return x[0].to_bytes(32, 'big') + x[1].to_bytes(32, 'big')
    return x.to_bytes(32, 'big')

def deserialize(x, is_ext=False):
    if is_ext:
        return (int.from_bytes(x[:32], 'big'), int.from_bytes(x[32:], 'big'))
    return int.from_bytes(x, 'big')

def ext_get_power_cycle(r, f, modulus):
    o = [f.one()]
    while o[-1] != f.one() or len(o) == 1:
        o.append(f.mul(o[-1], r))
    return o[:-1]

def ext_eval_quartic(f, p, x):
    xsq = f.mul(x, x)
    xcb = f.mul(xsq, x)
    return f.add(p[0], f.add(f.mul(p[1], x), f.add(f.mul(p[2], xsq), 
f.mul(p[3], xcb))))

def ext_zpoly(f, xs):
    root = [f.one()]
    for x in xs:
        root.insert(0, f.zero())
        for j in range(len(root)-1):
            root[j] = f.sub(root[j], f.mul(root[j+1], x))
    return root

def ext_div_polys(f, a, b):
    a = [x for x in a]
    o = []
    apos = len(a) - 1
    bpos = len(b) - 1
    diff = apos - bpos
    while diff >= 0:
        quot = f.div(a[apos], b[bpos])
        o.insert(0, quot)
        for i in range(bpos, -1, -1):
            a[diff+i] = f.sub(a[diff+i], f.mul(b[i], quot))
        apos -= 1
        diff -= 1
    return o

def ext_eval_poly_at(f, p, x):
    y = f.zero()
    for p_coeff in reversed(p):
        y = f.add(f.mul(y, x), p_coeff)
    return y

def ext_lagrange_interp(f, xs, ys):
    root = ext_zpoly(f, xs)
    nums = [ext_div_polys(f, root, [f.sub(f.zero(), x), f.one()]) for x in xs]
    denoms = [ext_eval_poly_at(f, nums[i], xs[i]) for i in range(len(xs))]
    invdenoms = f.multi_inv(denoms)
    b = [f.zero() for _ in ys]
    for i in range(len(xs)):
        yslice = f.mul(ys[i], invdenoms[i])
        for j in range(len(ys)):
            if nums[i][j] != f.zero() and yslice != f.zero():
                b[j] = f.add(b[j], f.mul(nums[i][j], yslice))
    return b

def ext_multi_interp_4(f, xsets, ysets):
    return [ext_lagrange_interp(f, xs, ys) for xs, ys in zip(xsets, ysets)]

# Generate an FRI proof that the polynomial that has the specified
# values at successive powers of the specified root of unity has a
# degree lower than maxdeg_plus_1
#
# We use maxdeg+1 instead of maxdeg because it's more mathematically
# convenient in this case.

def prove_low_degree(values, root_of_unity, maxdeg_plus_1, modulus, 
exclude_multiples_of=0):
    is_ext = isinstance(root_of_unity, tuple)
    f = ExtensionField(modulus) if is_ext else PrimeField(modulus)
    print('Proving %d values are degree <= %d' % (len(values), maxdeg_plus_1))

    # If the degree we are checking for is less than or equal to 32,
    # use the polynomial directly as a proof
    if maxdeg_plus_1 <= 16:
        print('Produced FRI proof')
        return [[serialize(x) for x in values]]

    # Calculate the set of x coordinates
    if is_ext:
        xs = ext_get_power_cycle(root_of_unity, f, modulus)
    else:
        xs = orig_get_power_cycle(root_of_unity, modulus)
    assert len(values) == len(xs)

    # Put the values into a Merkle tree. This is the root that the
    # proof will be checked against
    m = merkelize([serialize(v) for v in values])

    # Select a pseudo-random x coordinate
    special_x = int.from_bytes(m[1], 'big') % modulus
    if is_ext:
        special_x = (special_x, 0)

    # Calculate the "column" at that x coordinate
    # (see https://vitalik.ca/general/2017/11/22/starks_part_2.html)
    # We calculate the column by Lagrange-interpolating each row, and not
    # directly from the polynomial, as this is more efficient
    quarter_len = len(xs)//4
    x_subsets = [[xs[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)]
    y_subsets = [[values[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)]
    
    if is_ext:
        x_polys = ext_multi_interp_4(f, x_subsets, y_subsets)
    else:
        x_polys = f.multi_interp_4(x_subsets, y_subsets)

    if is_ext:
        column = [ext_eval_quartic(f, p, special_x) for p in x_polys]
    else:
        column = [f.eval_quartic(p, special_x) for p in x_polys]
        
    m2 = merkelize([serialize(c) for c in column])

    # Pseudo-randomly select y indices to sample
    ys = get_pseudorandom_indices(m2[1], len(column), 40, exclude_multiples_of=exclude_multiples_of)

    # Compute the positions for the values in the polynomial
    poly_positions = sum([[y + (len(xs) // 4) * j for j in range(4)] for y in ys], [])

    # This component of the proof, including Merkle branches
    o = [m2[1], mk_multi_branch(m2, ys), mk_multi_branch(m, poly_positions)]

    # Recurse...
    return [o] + prove_low_degree(column, f.exp(root_of_unity, 4),
                                  maxdeg_plus_1 // 4, modulus, 
exclude_multiples_of=exclude_multiples_of)

# Verify an FRI proof
def verify_low_degree_proof(merkle_root, root_of_unity, proof, maxdeg_plus_1, 
modulus, exclude_multiples_of=0):
    is_ext = isinstance(root_of_unity, tuple)
    f = ExtensionField(modulus) if is_ext else PrimeField(modulus)

    # Calculate which root of unity we're working with
    testval = root_of_unity
    roudeg = 1
    while testval != f.one():
        roudeg *= 2
        testval = f.mul(testval, testval)

    # Powers of the given root of unity 1, p, p**2, p**3 such that p**4 = 1
    quartic_roots_of_unity = [f.one(),
                              f.exp(root_of_unity, roudeg // 4),
                              f.exp(root_of_unity, roudeg // 2),
                              f.exp(root_of_unity, roudeg * 3 // 4)]

    # Verify the recursive components of the proof
    for prf in proof[:-1]:
        root2, column_branches, poly_branches = prf
        print('Verifying degree <= %d' % maxdeg_plus_1)

        # Calculate the pseudo-random x coordinate
        special_x = int.from_bytes(merkle_root, 'big') % modulus
        if is_ext:
            special_x = (special_x, 0)

        # Calculate the pseudo-randomly sampled y indices
        ys = get_pseudorandom_indices(root2, roudeg // 4, 40,
                                      
exclude_multiples_of=exclude_multiples_of)

        # Compute the positions for the values in the polynomial
        poly_positions = sum([[y + (roudeg // 4) * j for j in range(4)] for y 
in ys], [])

        # Verify Merkle branches
        column_values = verify_multi_branch(root2, ys, column_branches)
        poly_values = verify_multi_branch(merkle_root, poly_positions, 
poly_branches)

        # For each y coordinate, get the x coordinates on the row, the values on
        # the row, and the value at that y from the column
        xcoords = []
        rows = []
        columnvals = []
        for i, y in enumerate(ys):
            # The x coordinates from the polynomial
            x1 = f.exp(root_of_unity, y)
            xcoords.append([f.mul(quartic_roots_of_unity[j], x1) for j in 
range(4)])

            # The values from the original polynomial
            row = [deserialize(x, is_ext) for x in poly_values[i*4: i*4+4]]
            rows.append(row)

            columnvals.append(deserialize(column_values[i], is_ext))

        # Verify for each selected y coordinate that the four points from the
        # polynomial and the one point from the column that are on that y 
        # coordinate are on the same deg < 4 polynomial
        if is_ext:
            polys = ext_multi_interp_4(f, xcoords, rows)
        else:
            polys = f.multi_interp_4(xcoords, rows)

        for p, c in zip(polys, columnvals):
            if is_ext:
                assert ext_eval_quartic(f, p, special_x) == c
            else:
                assert f.eval_quartic(p, special_x) == c

        # Update constants to check the next proof
        merkle_root = root2
        root_of_unity = f.exp(root_of_unity, 4)
        maxdeg_plus_1 //= 4
        roudeg //= 4

    # Verify the direct components of the proof
    data = [deserialize(x, is_ext) for x in proof[-1]]
    print('Verifying degree <= %d' % maxdeg_plus_1)
    assert maxdeg_plus_1 <= 16

    # Check the Merkle root matches up
    mtree = merkelize([serialize(d) for d in data])
    assert mtree[1] == merkle_root

    # Check the degree of the data
    if is_ext:
        powers = ext_get_power_cycle(root_of_unity, f, modulus)
    else:
        powers = orig_get_power_cycle(root_of_unity, modulus)
        
    if exclude_multiples_of:
        pts = [x for x in range(len(data)) if x % exclude_multiples_of]
    else:
        pts = list(range(len(data)))

    if is_ext:
        poly = ext_lagrange_interp(f, [powers[x] for x in pts[:maxdeg_plus_1]],
                                   [data[x] for x in pts[:maxdeg_plus_1]])
    else:
        poly = f.lagrange_interp([powers[x] for x in pts[:maxdeg_plus_1]],
                                 [data[x] for x in pts[:maxdeg_plus_1]])
                                 
    for x in pts[maxdeg_plus_1:]:
        if is_ext:
            assert ext_eval_poly_at(f, poly, powers[x]) == data[x]
        else:
            assert f.eval_poly_at(poly, powers[x]) == data[x]

    print('FRI proof verified')
    return True