from permuted_tree import merkelize, mk_branch, verify_branch, blake, mk_multi_branch, verify_multi_branch
from poly_utils import PrimeField, ExtensionField
import time
import concurrent.futures
from fft import fft, bluestein_fft
from fri import prove_low_degree, verify_low_degree_proof, serialize
from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2
from poseidon1 import poseidon2 as poseidon1, ROUND_CONSTANTS, MATRIX_FULL, MATRIX_PARTIAL, DEFAULT_RF, DEFAULT_RP, DEFAULT_ALPHA, P, mat_vec_mul

# Single modulus for all operations (Mersenne prime)
modulus = 2**31 - 1

f = PrimeField(modulus)
f_ext = ExtensionField(modulus)
nonresidue = 3  # Better primitive root for this modulus than 7

spot_check_security_factor = 80
extension_factor = 4  # Now works with base 3 as primitive root

def _to_ext(a):
    """Convert integer to extension field element"""
    return a if isinstance(a, tuple) else (a, 0)

# Helper: Apply Poseidon2 permutation and return all 16 state elements
def poseidon2_full_state(scalar_val, steps):
    """Apply Poseidon2 permutation 'steps' times, starting with scalar in first state element.
    Returns the full 16-element state after all steps.
    All operations in koala modulus."""
    state = [scalar_val % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2

    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        
        if round_idx == 0:
            rc_idx = 0

        is_full = True if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else False
        
        if is_full:
            # Apply round constants to all 16 elements
            for j in range(16):
                state[j] = (state[j] + ROUND_CONSTANTS[rc_idx]) % modulus
                rc_idx += 1
            for j in range(16):
                state[j] = pow(state[j], DEFAULT_ALPHA, modulus)
            state = mat_vec_mul(MATRIX_FULL, state, modulus)
        else:
            # Apply round constant ONLY to first element
            state[0] = (state[0] + ROUND_CONSTANTS[rc_idx]) % modulus
            rc_idx += 1
            state[0] = pow(state[0], DEFAULT_ALPHA, modulus)
            state = mat_vec_mul(MATRIX_PARTIAL, state, modulus)
            
    return state

# Generate a STARK for a Poseidon2 calculation
def mk_poseidon2_proof(inp, steps, verbose_timing=False, track_fft=False):
    start_time = time.time()
    timings = {} if verbose_timing else None
    phase_start = start_time
    fft_stats = {'operations': [], 'total_ops': 0} if track_fft else None

    # Some constraints to make our job easier
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps)

    # Total Size of the evaluation domain
    precision = steps * extension_factor

    # Root of unity such that x^precision=1 in F_p^2
    G2 = f_ext.exp((nonresidue, 1), (modulus**2 - 1) // precision)

    # Root of unity such that x^steps=1
    skips = precision // steps
    G1 = f_ext.exp(G2, skips)
    G1_inv = f_ext.inv(G1)

    # Powers of the higher-order root of unity
    xs = [f_ext.exp(G2, i) for i in range(precision)]
    last_step_position = xs[(steps-1)*extension_factor]

    # Generate public polynomials for selectors and round constants
    s_full_trace = []
    rc_traces = [[] for _ in range(16)]
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        is_full_val = (1, 0) if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else (0, 0)
        s_full_trace.append(is_full_val)
        for j in range(16):
            if is_full_val == (1, 0):
                rc_traces[j].append((ROUND_CONSTANTS[rc_idx], 0))
                rc_idx += 1
            else:
                if j == 0:
                    rc_traces[j].append((ROUND_CONSTANTS[rc_idx], 0))
                    rc_idx += 1
                else:
                    rc_traces[j].append((0, 0))
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        s_inv_future = executor.submit(fft, s_full_trace, modulus, G1, True, f_ext)
        rc_inv_futures = [executor.submit(fft, rc_traces[j], modulus, G1, True, f_ext) for j in range(16)]
        
        s_full_poly = s_inv_future.result()
        rc_polys = [f.result() for f in rc_inv_futures]

        s_fwd_future = executor.submit(fft, s_full_poly, modulus, G2, False, f_ext)
        rc_fwd_futures = [executor.submit(fft, rc_polys[j], modulus, G2, False, f_ext) for j in range(16)]
        
        s_full_evals = s_fwd_future.result()
        rc_evals = [f.result() for f in rc_fwd_futures]

    if track_fft:
        fft_stats['operations'].extend([
            {'phase': 'selector_rc', 'type': 'inverse', 'size': len(s_full_trace), 'count': 1},
            {'phase': 'selector_rc', 'type': 'inverse', 'size': len(rc_traces[0]), 'count': 16},
            {'phase': 'selector_rc', 'type': 'forward', 'size': len(s_full_poly), 'count': 1},
            {'phase': 'selector_rc', 'type': 'forward', 'size': len(rc_polys[0]), 'count': 16},
        ])
        fft_stats['total_ops'] += 34
    if verbose_timing:
        timings['selector_rc_poly'] = time.time() - phase_start
        phase_start = time.time()
    print('Generated Selector and Round Constant polynomials (Public)')

    # Generate the computational trace: 16 x steps matrix
    # trace[step_index][state_element_index] = value of that state element at that step
    # step i corresponds to the state after applying Poseidon2 permutation i times
    computational_trace = []
    current_state = [inp % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    computational_trace.append(list(current_state))  # step 0: initial state (0 applications)
    
    for i in range(steps - 1):  # Apply permutation steps-1 times to get steps entries total
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        
        current_state = [(current_state[j] + rc_traces[j][i][0]) % modulus for j in range(16)]
        if is_full:
            current_state = [pow(x, DEFAULT_ALPHA, modulus) for x in current_state]
            current_state = [sum(MATRIX_FULL[r][c] * current_state[c] for c in range(16)) % modulus for r in range(16)]
        else:
            current_state[0] = pow(current_state[0], DEFAULT_ALPHA, modulus)
            current_state = [sum(MATRIX_PARTIAL[r][c] * current_state[c] for c in range(16)) % modulus for r in range(16)]
            
        computational_trace.append(current_state)
    
    # Extract input and output states
    input_state = computational_trace[0]
    
    valid_steps = ((steps - 1) // rounds_per_hash) * rounds_per_hash
    output_state = computational_trace[valid_steps]

    if verbose_timing:
        timings['computational_trace'] = time.time() - phase_start
        phase_start = time.time()
    print('Done generating poseidon computational trace (16-column)')

    # Interpolate each of the 16 state element columns into a polynomial
    # P_j(x) is the polynomial whose evaluations at powers of G1 are the j-th state element across steps
    p_polynomials = []
    p_evaluations_list = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Interpolate into polynomials
        p_inv_futures = [executor.submit(fft, [(computational_trace[i][j], 0) for i in range(steps)], modulus, G1, True, f_ext) for j in range(16)]
        p_polynomials = [f.result() for f in p_inv_futures]
        
        # Low-degree extend over the larger domain G2
        p_fwd_futures = [executor.submit(fft, p_polynomials[j], modulus, G2, False, f_ext) for j in range(16)]
        p_evaluations_list = [f.result() for f in p_fwd_futures]

    if track_fft:
        fft_stats['operations'].extend([
            {'phase': 'trace_interpolation', 'type': 'inverse', 'size': steps, 'count': 16},
            {'phase': 'trace_interpolation', 'type': 'forward', 'size': steps, 'count': 16},
        ])
        fft_stats['total_ops'] += 32
    if verbose_timing:
        timings['trace_interpolation'] = time.time() - phase_start
        phase_start = time.time()
    print('Converted 16-column computational trace into polynomials and low-degree extended them')

    # Compute the 16 composed polynomials C_j
    c_evaluations_list = [[] for _ in range(16)]

    for i in range(precision):
        s_f = s_full_evals[i]
        
        state_plus_rc = [f_ext.add(p_evaluations_list[k][i], rc_evals[k][i]) for k in range(16)]
        s_f_inv = f_ext.sub(f_ext.one(), s_f)
        sbox_state_0 = f_ext.exp(state_plus_rc[0], DEFAULT_ALPHA)
        sbox_state_others = [
            f_ext.add(
                f_ext.mul(f_ext.exp(state_plus_rc[k], DEFAULT_ALPHA), s_f),
                f_ext.mul(state_plus_rc[k], s_f_inv)
            ) for k in range(1, 16)]
        sbox_state = [sbox_state_0] + sbox_state_others
        
        if i == 50:
            print(f"DEBUG Prover at i=50: s_f={s_f}, state_plus_rc[0]={state_plus_rc[0]}, sbox_state[0]={sbox_state_0}")
        
        for j in range(16):
            mat_full_res = f_ext.zero()
            mat_partial_res = f_ext.zero()
            for k in range(16):
                mf_val = (MATRIX_FULL[j][k], 0)
                mp_val = (MATRIX_PARTIAL[j][k], 0)
                mat_full_res = f_ext.add(mat_full_res, f_ext.mul(sbox_state[k], mf_val))
                mat_partial_res = f_ext.add(mat_partial_res, f_ext.mul(sbox_state[k], mp_val))
            
            p_next_calculated = f_ext.add(
                f_ext.mul(mat_full_res, s_f),
                f_ext.mul(mat_partial_res, s_f_inv)
            )

            next_p_j = p_evaluations_list[j][(i + extension_factor) % precision]
            c_evaluations_list[j].append(f_ext.sub(next_p_j, p_next_calculated))

    if verbose_timing:
        timings['transition_constraints'] = time.time() - phase_start
        phase_start = time.time()
    print('Computed 16 transition constraint polynomials')

    # Compute D(x) = (sum of C_j(x) weighted by random coefficients) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    # First, generate random coefficients for combining the 16 C polynomials
    # We'll compute this after we have a Merkle root to derive randomness
    # For now, compute the basis: numerator of Z(x) and its inverses
    
    last_step_position = xs[(steps-1)*extension_factor]
    z_num_evaluations = [f_ext.sub(xs[(i * steps) % precision], f_ext.one()) for i in range(precision)]
    z_num_inv = f_ext.multi_inv(z_num_evaluations)
    z_den_evaluations = [f_ext.sub(xs[i], last_step_position) for i in range(precision)]
    
    # Compute the B polynomial using all 16 state elements at boundaries
    # For each state element j, we need: B_j(x) * Q(x) + I_j(x) = P_j(x)
    # where I_j interpolates input_state[j] at x=1 and output_state[j] at x=x_atlast_step
    
    b_evaluations_list = []
    
    valid_step_position = xs[valid_steps * extension_factor]

    # (x−1)(x−valid_step_position) as extension-field polynomial
    zeropoly2 = f_ext.mul_polys([((-1, 0)), ((1, 0))], [f_ext.sub(f_ext.zero(), valid_step_position), f_ext.one()])
    # 1 / (Q(G2^i)) for all i (extension field)
    inv_z2_evaluations = f_ext.multi_inv([f_ext.eval_poly_at(zeropoly2, x) for x in xs])

    for j in range(16):
        # Interpolate boundaries for state element j in 2 Points
        # Point 1:(x,y)= (1, input_state[j])
        # Point 2:(x,y)= (valid_step_position, output_state[j])
        boundary_xs = [f_ext.one(), valid_step_position]
        boundary_ys = [_to_ext(input_state[j]), _to_ext(output_state[j])]
        # interpolant_j  = [c0,c1] (degree 1 Polynomial I_j(x))
        interpolant_j = f_ext.lagrange_interp_2(boundary_xs, boundary_ys)
        # Evaluates I_j(x) at all precision points
        i_j_evaluations = [f_ext.eval_poly_at(interpolant_j, x) for x in xs]
        # Compute B_j(x) = (P_j(x) - I_j(x)) / Q(x)
        b_j_evals = [f_ext.mul(f_ext.sub(p_evaluations_list[j][i], i_j_evaluations[i]), inv_z2_evaluations[i])
                     for i in range(precision)]
        b_evaluations_list.append(b_j_evals)

    if verbose_timing:
        timings['boundary_constraints'] = time.time() - phase_start
        phase_start = time.time()
    print('Computed 16 boundary constraint polynomials (B)')

    # Compute D and B via weighted combination of their 16 components
    # Derive random coefficients from input value and steps (deterministic for prover and verifier)
    # Random Weights for each B_j and C_j
    k_c = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    
    # Convert k_c and k_b to extension field
    k_c_ext = [_to_ext(k) for k in k_c]
    k_b_ext = [_to_ext(k) for k in k_b]
    
    # Compute D(x) = (Σ C_j(x) * k_c[j]) / Z(x) at all evaluation points
    # Z(x) = (x^steps - 1) / (x - last_step_position)
    # At each evaluation point x_i = G2^i:
    #   D(x_i) = (Σ C_j(x_i) * k_c[j]) * (x_i - last) / ((x_i)^steps - 1)
    d_evaluations = [f_ext.zero()] * precision
    for i in range(precision):
        # Numerator
        c_sum = f_ext.zero()
        for j in range(16):
            c_sum = f_ext.add(c_sum, f_ext.mul(c_evaluations_list[j][i], k_c_ext[j]))
        # Denominator: divide by Z(x) = (x^steps - 1) / (x - last_step_position)
        # Computing as: c_sum * (x - last) / (x^steps - 1)
        d_evaluations[i] = f_ext.mul(f_ext.mul(c_sum, z_den_evaluations[i]), z_num_inv[i])
    
    # Compute B_combined(x) = sum of B_j(x) * k_b[j]
    b_evaluations = [f_ext.zero()] * precision
    for i in range(precision):
        b_sum = f_ext.zero()
        for j in range(16):
            b_sum = f_ext.add(b_sum, f_ext.mul(b_evaluations_list[j][i], k_b_ext[j]))
        b_evaluations[i] = b_sum

    if verbose_timing:
        timings['weighted_combination'] = time.time() - phase_start
        phase_start = time.time()
    print('Computed D and B polynomials via weighted combination')
    
    # Now build the final Merkle tree with all components: 16 P + B + D
    # Store first leaf's D value for debugging
    test_d = d_evaluations[50]
    print(f"DEBUG Prover: d_evaluations[50]={test_d}, d_evaluations[54]={d_evaluations[54]}")
    print(f"DEBUG Prover: P[0][50]={p_evaluations_list[0][50]}, P[0][54]={p_evaluations_list[0][54]}")
    print(f"DEBUG Prover: xs[50]={xs[50]}, xs[54]={xs[54]}")
    
    mtree = merkelize([b''.join(serialize(p_evaluations_list[j][i]) for j in range(16)) +
                       serialize(b_evaluations[i]) +
                       serialize(d_evaluations[i])
                       for i in range(precision)])
    if verbose_timing:
        timings['merkle_tree'] = time.time() - phase_start
        phase_start = time.time()
    print('Computed hash root')

    # Based on the hashes of P, D and B, we select a random linear combination
    # of all 16 P columns, P*x^steps, B, B*x^steps, and D, and prove the low-degreeness
    k1 = int.from_bytes(blake(mtree[1] + b'\x01'), 'big') % modulus
    k2 = int.from_bytes(blake(mtree[1] + b'\x02'), 'big') % modulus
    k3 = int.from_bytes(blake(mtree[1] + b'\x03'), 'big') % modulus
    k4 = int.from_bytes(blake(mtree[1] + b'\x04'), 'big') % modulus

    # Individual weights for each of the 16 P columns
    k_p = [int.from_bytes(blake(mtree[1] + bytes([10+j])), 'big') % modulus for j in range(16)]

    # Compute the linear combination of all polynomials
    G2_to_the_steps = f_ext.exp(G2, steps)
    powers = [f_ext.one()]
    for i in range(1, precision):
        powers.append(f_ext.mul(powers[-1], G2_to_the_steps))

    l_evaluations = [f_ext.zero()] * precision
    for i in range(precision):
        # Add weighted sum of all 16 P columns
        p_sum = f_ext.zero()
        for j in range(16):
            p_sum = f_ext.add(p_sum, f_ext.mul(p_evaluations_list[j][i], _to_ext(k_p[j])))
        
        p_sum_weighted = f_ext.add(
            f_ext.mul(p_sum, _to_ext(k1)),
            f_ext.mul(f_ext.mul(p_sum, _to_ext(k2)), powers[i])
        )
        
        # Add B and D terms
        l_evaluations[i] = f_ext.add(
            f_ext.add(d_evaluations[i], p_sum_weighted),
            f_ext.add(
                f_ext.mul(b_evaluations[i], _to_ext(k3)),
                f_ext.mul(f_ext.mul(b_evaluations[i], powers[i]), _to_ext(k4))
            )
        )

    l_mtree = merkelize([serialize(val) for val in l_evaluations])
    if verbose_timing:
        timings['linear_combination'] = time.time() - phase_start
        phase_start = time.time()
    print('Computed random linear combination')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates, excluding
    # multiples of `extension_factor`
    branches = []
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + extension_factor) % precision] for x in positions], [])
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    fri_start = time.time() if verbose_timing else None
    fri_proof = prove_low_degree(l_evaluations, G2, steps * 4, modulus, exclude_multiples_of=extension_factor)
    if verbose_timing:
        timings['fri_proof'] = time.time() - fri_start

    o = [mtree[1],
         l_mtree[1],
         mk_multi_branch(mtree, curr_next_positions),
         mk_multi_branch(l_mtree, positions),
         fri_proof]

    total_time = time.time() - start_time
    if verbose_timing:
        timings['total'] = total_time
        if track_fft:
            return o, timings, fft_stats
        return o, timings
    if track_fft:
        return o, fft_stats
    print("Poseidon2 STARK computed in %.4f sec" % total_time)
    return o

# Verifies a Poseidon2 STARK
def verify_poseidon2_proof(inp, steps, output, proof):
    import time
    from utils import get_pseudorandom_indices, is_a_power_of_2
    from fri import serialize
    
    start_time = time.time()
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    # Reconstruct all values needed for verification (same as proof generation)
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps)
    
    precision = steps * extension_factor
    valid_steps = ((steps - 1) // rounds_per_hash) * rounds_per_hash
    
    # Root of unity such that x^precision=1 in F_p^2 (EXTENSION FIELD!)
    G2 = f_ext.exp((nonresidue, 1), (modulus**2 - 1) // precision)
    
    # Root of unity such that x^steps=1
    skips = precision // steps
    G1 = f_ext.exp(G2, skips)
    
    # Powers of the higher-order root of unity
    xs = [f_ext.exp(G2, i) for i in range(precision)]
    last_step_position = xs[(steps-1)*extension_factor]
    valid_step_position = xs[valid_steps*extension_factor]
    
    # Regenerate public polynomials for selectors and round constants
    s_full_trace = []
    rc_traces = [[] for _ in range(16)]
    
    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        s_full_trace.append((is_full, 0))  # As extension field element
        for j in range(16):
            if is_full:
                rc_traces[j].append((ROUND_CONSTANTS[rc_idx], 0))
                rc_idx += 1
            else:
                if j == 0:
                    rc_traces[j].append((ROUND_CONSTANTS[rc_idx], 0))
                    rc_idx += 1
                else:
                    rc_traces[j].append((0, 0))
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    # FFT with extension field
    s_full_poly = fft(s_full_trace, modulus, G1, inv=True, field=f_ext)
    rc_polys = [fft(rc_traces[j], modulus, G1, inv=True, field=f_ext) for j in range(16)]
    
    # Forward FFT to get evaluations
    s_full_evals = fft(s_full_poly, modulus, G2, False, f_ext)
    rc_evals = [fft(rc_polys[j], modulus, G2, False, f_ext) for j in range(16)]
    
    # Extract proof components
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    
    # Verify the FRI proof first
    assert verify_low_degree_proof(l_root, G2, fri_proof, steps * 4, modulus, exclude_multiples_of=extension_factor)
    print('FRI proof verified')
    
    # Derive random coefficients deterministically (same as prover)
    k_c = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    k_c_ext = [_to_ext(k) for k in k_c]
    k_b_ext = [_to_ext(k) for k in k_b]
    
    # Coefficients for linear combination
    k1 = int.from_bytes(blake(m_root + b'\x01'), 'big') % modulus
    k2 = int.from_bytes(blake(m_root + b'\x02'), 'big') % modulus
    k3 = int.from_bytes(blake(m_root + b'\x03'), 'big') % modulus
    k4 = int.from_bytes(blake(m_root + b'\x04'), 'big') % modulus
    k_p = [int.from_bytes(blake(m_root + bytes([10+j])), 'big') % modulus for j in range(16)]
    k_p_ext = [_to_ext(k) for k in k_p]
    k1_ext = _to_ext(k1)
    k2_ext = _to_ext(k2)
    k3_ext = _to_ext(k3)
    k4_ext = _to_ext(k4)
    
    # Get pseudorandom sample positions for spot checks
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + extension_factor) % precision] for x in positions], [])
    
    print(f"DEBUG: positions={positions[:10]}, curr_next={curr_next_positions[:20]}")
    
    # Get merkle branches
    main_branch_leaves = verify_multi_branch(m_root, curr_next_positions, main_branches)
    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)
    
    # Compute the reference Poseidon2 state at valid_steps for boundary check
    reference_state = [inp % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(valid_steps):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        
        reference_state = [(reference_state[j] + rc_traces[j][i][0]) % modulus for j in range(16)]
        if is_full:
            reference_state = [pow(x, DEFAULT_ALPHA, modulus) for x in reference_state]
            reference_state = [sum(MATRIX_FULL[r][c] * reference_state[c] for c in range(16)) % modulus for r in range(16)]
        else:
            reference_state[0] = pow(reference_state[0], DEFAULT_ALPHA, modulus)
            reference_state = [sum(MATRIX_PARTIAL[r][c] * reference_state[c] for c in range(16)) % modulus for r in range(16)]
    
    assert reference_state[0] == output, f"Output mismatch: expected {output}, got {reference_state[0]}"
    
    # Verify spot checks
    for i, pos in enumerate(positions):
        x = xs[pos]
        x_to_the_steps = f_ext.exp(x, steps)
        
        # Get merkle leaves at current and next positions
        mbranch_curr = main_branch_leaves[i*2]
        mbranch_next = main_branch_leaves[i*2+1]
        l_leaf = linear_comb_branch_leaves[i]
        
        next_pos = (pos + extension_factor) % precision
        if i == 0:
            print(f"DEBUG: i={i}, pos={pos}, next_pos={next_pos}, curr_next[{i*2}]={curr_next_positions[i*2]}, curr_next[{i*2+1}]={curr_next_positions[i*2+1]}")
            print(f"DEBUG: verifier xs[{pos}]={xs[pos]}, xs[{next_pos}]={xs[next_pos]}")
        
        # Deserialize P_j, B, D values (extension field tuples)
        p_of_x = []
        for j in range(16):
            byte_offset = j * 64
            a = int.from_bytes(mbranch_curr[byte_offset:byte_offset+32], 'big')
            b = int.from_bytes(mbranch_curr[byte_offset+32:byte_offset+64], 'big')
            p_of_x.append((a, b))
        
        if i == 0:
            print(f"DEBUG: p_of_x[0]={p_of_x[0]}")
        
        byte_offset = 16 * 64
        a = int.from_bytes(mbranch_curr[byte_offset:byte_offset+32], 'big')
        b = int.from_bytes(mbranch_curr[byte_offset+32:byte_offset+64], 'big')
        b_of_x = (a, b)
        
        byte_offset = 17 * 64
        a = int.from_bytes(mbranch_curr[byte_offset:byte_offset+32], 'big')
        b = int.from_bytes(mbranch_curr[byte_offset+32:byte_offset+64], 'big')
        d_of_x = (a, b)
        
        if i == 0:
            print(f"DEBUG: d_leaf_bytes at offset {byte_offset}={mbranch_curr[byte_offset:byte_offset+64].hex()}")
        
        # Get P_j(g1*x) from next leaf
        p_of_next = []
        for j in range(16):
            byte_offset = j * 64
            a = int.from_bytes(mbranch_next[byte_offset:byte_offset+32], 'big')
            b = int.from_bytes(mbranch_next[byte_offset+32:byte_offset+64], 'big')
            p_of_next.append((a, b))
        
        if i == 0:
            print(f"DEBUG: p_of_next[0]={p_of_next[0]}")
        
        # Deserialize L(x)
        a = int.from_bytes(l_leaf[:32], 'big')
        b = int.from_bytes(l_leaf[32:64], 'big')
        l_of_x = (a, b)
        
        # Get selector and round constants at x
        s_f = f_ext.eval_poly_at(s_full_poly, x)
        rcs = [f_ext.eval_poly_at(rc_polys[j], x) for j in range(16)]
        
        # Apply Poseidon2 transition logic in extension field
        state_plus_rc = [f_ext.add(p_of_x[j], rcs[j]) for j in range(16)]
        sbox_state = []
        for j in range(16):
            val = state_plus_rc[j]
            # All elements go through sbox in full round, only [0] in partial
            if j == 0:
                sbox_state.append(f_ext.exp(val, DEFAULT_ALPHA))
            else:
                full_sbox = f_ext.exp(val, DEFAULT_ALPHA)
                partial_identity = val
                sbox_state.append(f_ext.add(f_ext.mul(full_sbox, s_f), 
                                            f_ext.mul(partial_identity, f_ext.sub(f_ext.one(), s_f))))
        
        # Matrix multiply in extension field
        mat_full_res = []
        mat_partial_res = []
        for row in range(16):
            full_acc = f_ext.zero()
            partial_acc = f_ext.zero()
            for col in range(16):
                coeff_full = MATRIX_FULL[row][col]
                coeff_partial = MATRIX_PARTIAL[row][col]
                full_acc = f_ext.add(full_acc, f_ext.mul(sbox_state[col], _to_ext(coeff_full)))
                partial_acc = f_ext.add(partial_acc, f_ext.mul(sbox_state[col], _to_ext(coeff_partial)))
            mat_full_res.append(full_acc)
            mat_partial_res.append(partial_acc)
        
        # Compute transition constraints: C_j = P_j(g1*x) - M(sbox(P_j(x) + RC))
        # where M is either full or partial matrix depending on s_f
        d_sum = f_ext.zero()
        for j in range(16):
            p_next_expected = f_ext.add(f_ext.mul(mat_full_res[j], s_f),
                                        f_ext.mul(mat_partial_res[j], f_ext.sub(f_ext.one(), s_f)))
            c_j = f_ext.sub(p_of_next[j], p_next_expected)
            d_sum = f_ext.add(d_sum, f_ext.mul(c_j, k_c_ext[j]))
        
        if i == 0:
            print(f"DEBUG: c_j[0]={f_ext.sub(p_of_next[0], f_ext.add(f_ext.mul(mat_full_res[0], s_f), f_ext.mul(mat_partial_res[0], f_ext.sub(f_ext.one(), s_f))))}")
            print(f"DEBUG: d_sum={d_sum}")
        
        # Verify: D(x) = d_sum / Z(x) where Z(x) = (x^steps - 1) / (x - last_step)
        z_num = f_ext.sub(x_to_the_steps, f_ext.one())
        z_den = f_ext.sub(x, last_step_position)
        z_value = f_ext.div(z_num, z_den)
        expected_d = f_ext.mul(d_sum, f_ext.inv(z_value))
        
        if i == 0:
            print(f"DEBUG pos={pos}: d_of_x={d_of_x}, expected_d={expected_d}")
            print(f"  z_num={z_num}, z_den={z_den}, z_value={z_value}")
            print(f"  d_sum={d_sum}")
        
        assert d_of_x == expected_d, f"Transition constraint mismatch at position {pos}"
        
        # Verify boundary constraint: B(x) = weighted_sum(P_j(x) - I_j(x)) / Z2(x)
        # where Z2(x) = (x - 1) * (x - valid_step_position)
        z2_value = f_ext.mul(f_ext.sub(x, f_ext.one()), f_ext.sub(x, valid_step_position))
        b_sum = f_ext.zero()
        
        for j in range(16):
            # Interpolate boundary: I_j(1) = input[j], I_j(valid_step) = reference_state[j]
            input_val = _to_ext(inp) if j == 0 else f_ext.zero()
            output_val = _to_ext(reference_state[j])
            # Lagrange basis: (x - valid_step) / (1 - valid_step)
            l1 = f_ext.div(f_ext.sub(x, valid_step_position),
                          f_ext.sub(f_ext.one(), valid_step_position))
            # Lagrange basis: (x - 1) / (valid_step - 1)
            l2 = f_ext.div(f_ext.sub(x, f_ext.one()),
                          f_ext.sub(valid_step_position, f_ext.one()))
            i_j_x = f_ext.add(f_ext.mul(input_val, l1), f_ext.mul(output_val, l2))
            
            b_j = f_ext.sub(p_of_x[j], i_j_x)
            b_sum = f_ext.add(b_sum, f_ext.mul(b_j, k_b_ext[j]))
        
        expected_b = f_ext.mul(b_sum, f_ext.inv(z2_value))
        assert b_of_x == expected_b, f"Boundary constraint mismatch at position {pos}"
        
        # Verify linear combination: L(x) = weighted_sum(P_j, B, D)
        p_sum = f_ext.zero()
        for j in range(16):
            p_sum = f_ext.add(p_sum, f_ext.mul(p_of_x[j], k_p_ext[j]))
        
        expected_l = f_ext.add(d_of_x, f_ext.add(f_ext.mul(p_sum, k1_ext),
                              f_ext.add(f_ext.mul(f_ext.mul(p_sum, x_to_the_steps), k2_ext),
                              f_ext.add(f_ext.mul(b_of_x, k3_ext),
                                       f_ext.mul(f_ext.mul(b_of_x, x_to_the_steps), k4_ext)))))
        
        assert l_of_x == expected_l, f"Linear combination mismatch at position {pos}"
    
    elapsed = time.time() - start_time
    print(f'Verified {spot_check_security_factor} consistency checks')
    print(f'Verified Poseidon2 STARK in {elapsed:.4f} sec')
    return True
