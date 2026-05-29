from permuted_tree import merkelize, mk_branch, verify_branch, blake, mk_multi_branch, verify_multi_branch
from poly_utils import PrimeField
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from mixed_radix import fft
from fri import prove_low_degree, verify_low_degree_proof
from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2
from poseidon1 import poseidon1, ROUND_CONSTANTS, MATRIX_FULL, DEFAULT_RF, DEFAULT_RP, DEFAULT_ALPHA, P, mat_vec_mul

# Single modulus for all operations
modulus = 2**24 *127 + 1

f = PrimeField(modulus)
nonresidue = 3  # Better primitive root for this modulus than 7

spot_check_security_factor = 80
extension_factor = 4

print("[poseidon_stark] Module loaded successfully")

# Helper: Apply Poseidon permutation and return all 16 state elements
def poseidon_full_state(scalar_val, steps):
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
            for j in range(16):
                state[j] = (state[j] + ROUND_CONSTANTS[rc_idx]) % modulus
                rc_idx += 1
            for j in range(16):
                state[j] = pow(state[j], DEFAULT_ALPHA, modulus)
        else:
            state[0] = (state[0] + ROUND_CONSTANTS[rc_idx]) % modulus
            rc_idx += 1
            state[0] = pow(state[0], DEFAULT_ALPHA, modulus)
        state = mat_vec_mul(MATRIX_FULL, state, modulus)
    return state


# Generate a STARK for a Poseidon calculation
def mk_poseidon_proof(inp, steps):
    start_time = time.time()
    print(f"[mk_poseidon_proof] Starting with inp={inp}, steps={steps}")
    
    # OPTION 1: Transform steps to mixed-radix form 2^k × 127
    # This embeds the 127 factor into steps, fully utilizing (p-1) = 2^24 × 127
    # For input steps = 2^LOGSTEPS, compute: steps = 2^(LOGSTEPS-7) × 127
    #
    # Examples:
    #   LOGSTEPS=2  (steps=4)   → 2^0 × 127 = 127
    #   LOGSTEPS=10 (steps=1024) → 2^3 × 127 = 1,016
    #   LOGSTEPS=11 (steps=2048) → 2^4 × 127 = 2,032
    
    # Calculate the power-of-2 component
    logsteps_input = steps.bit_length() - 1 if steps > 0 else 0
    logsteps_2radix = max(0, logsteps_input - 7)
    
    # Compute mixed-radix steps
    steps = (2 ** logsteps_2radix) * 127
    
    # Updated constraints for mixed-radix form
    assert steps <= 2**24 * 127, f"steps {steps} exceeds maximum domain (2^24 × 127)"
    assert (modulus - 1) % steps == 0, f"steps {steps} must divide (p-1)={modulus-1}"

    # Total Size of the evaluation domain
    precision = steps * extension_factor

    # Root of unity such that x^precision=1
    G2 = f.exp(nonresidue, (modulus-1)//precision)

    # Root of unity such that x^steps=1
    skips = precision // steps
    G1 = f.exp(G2, skips)

    # Compute powers of G2 only as needed (don't generate full cycle)
    # We need powers 0 through precision
    # Build efficiently using multiplication instead of repeated exponentiation
    max_index = max((steps - 1) * extension_factor, precision)
    xs = [1]  # G2^0 = 1
    current = G2
    for i in range(1, max_index + 1):
        xs.append(current)
        current = (current * G2) % modulus
    last_step_position = xs[(steps - 1) * extension_factor]
    print(f"[mk_poseidon_proof] Computed roots of unity, xs computed")

    # Generate public polynomials for selectors and round constants
    print(f"[mk_poseidon_proof] Generating selector and round constant traces...")
    s_full_trace = []
    rc_traces = [[] for _ in range(16)]
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        s_full_trace.append(is_full)
        if is_full:
            for j in range(16):
                rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                rc_idx += 1
        else:
            rc_traces[0].append(ROUND_CONSTANTS[rc_idx])
            rc_idx += 1
            for j in range(1, 16):
                rc_traces[j].append(0)
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    print(f"[mk_poseidon_proof] Traces generated, starting all FFTs in single pool...")
    # === Single consolidated ProcessPoolExecutor for ALL FFT operations ===
    # (Avoids 3× pool spawn/join overhead — saves ~5s)
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Phase 1: Inverse FFT of selector + 16 round constant traces (17 tasks)
        s_inv_future = executor.submit(fft, s_full_trace, modulus, G1, True)
        rc_inv_futures = [executor.submit(fft, rc_traces[j], modulus, G1, True) for j in range(16)]
        s_full_poly = s_inv_future.result()
        rc_polys = [fut.result() for fut in rc_inv_futures]
        print(f"[mk_poseidon_proof] Got s_full_poly and rc_polys")
        
        # Phase 1b: Forward FFT (LDE) of selector and RC polys (17 tasks)
        s_full_padded = s_full_poly + [0] * (precision - len(s_full_poly))
        s_fwd_future = executor.submit(fft, s_full_padded, modulus, G2, False)
        rc_fwd_futures = []
        for j in range(16):
            rc_padded = rc_polys[j] + [0] * (precision - len(rc_polys[j]))
            rc_fwd_futures.append(executor.submit(fft, rc_padded, modulus, G2, False))
        s_full_evals = s_fwd_future.result()
        rc_evals = [fut.result() for fut in rc_fwd_futures]
    
    print('Generated Selector and Round Constant polynomials (Public)')

    # Generate the computational trace: 16 state + 16 auxiliary (w = (state+rc)²)
    # trace[step_index][state_element_index] = value of that state element at that step
    computational_trace = []
    w_trace = []  # Auxiliary columns: w[j] = (state[j] + rc[j])²
    current_state = [inp % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    computational_trace.append(list(current_state))
    
    for i in range(steps - 1):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        
        state_plus_rc = [(current_state[j] + rc_traces[j][i]) % modulus for j in range(16)]
        
        # Compute w = (state + rc)² for the auxiliary trace
        # For full rounds: all 16 elements get S-box, so all w_j are needed
        # For partial rounds: only w_0 is needed, but we compute all for uniformity
        w_values = [pow(state_plus_rc[j], 2, modulus) for j in range(16)]
        w_trace.append(w_values)
        
        if is_full:
            # S-box: (state+rc)^3 = w * (state+rc)
            sbox_out = [(w_values[j] * state_plus_rc[j]) % modulus for j in range(16)]
            current_state = [sum(MATRIX_FULL[r][c] * sbox_out[c] for c in range(16)) % modulus for r in range(16)]
        else:
            # Partial round: S-box only on state[0]
            sbox0 = (w_values[0] * state_plus_rc[0]) % modulus
            partial_state = [sbox0] + state_plus_rc[1:]
            current_state = [sum(MATRIX_FULL[r][c] * partial_state[c] for c in range(16)) % modulus for r in range(16)]
            
        computational_trace.append(current_state)
    
    # Add a final w_trace entry for the last step (needed for trace length matching)
    # Use zeros since there's no constraint on the last step
    w_trace.append([0] * 16)
    
    input_state = computational_trace[0]
    valid_steps = ((steps - 1) // rounds_per_hash) * rounds_per_hash
    output_state = computational_trace[valid_steps]
    print(f"DEBUG IN PROVER: valid_steps={valid_steps}, output_state[0]={output_state[0]}")
    
    print('Done generating poseidon computational trace (16 state + 16 aux columns)')

    # Interpolate each of the 16 state columns + 16 aux columns into polynomials
    # Then low-degree extend — all in the same pool
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Phase 2: Inverse FFT of 16 state + 16 aux traces (32 tasks)
        p_inv_futures = [executor.submit(fft, [computational_trace[i][j] for i in range(steps)], modulus, G1, True) for j in range(16)]
        w_inv_futures = [executor.submit(fft, [w_trace[i][j] for i in range(steps)], modulus, G1, True) for j in range(16)]
        p_polynomials = [fut.result() for fut in p_inv_futures]
        w_polynomials = [fut.result() for fut in w_inv_futures]
        
        # Phase 3: Forward FFT (LDE) of 32 traces (32 tasks) — same pool!
        p_fwd_futures = []
        w_fwd_futures = []
        for j in range(16):
            p_padded = p_polynomials[j] + [0] * (precision - len(p_polynomials[j]))
            p_fwd_futures.append(executor.submit(fft, p_padded, modulus, G2, False))
            w_padded = w_polynomials[j] + [0] * (precision - len(w_polynomials[j]))
            w_fwd_futures.append(executor.submit(fft, w_padded, modulus, G2, False))
        p_evaluations_list = [fut.result() for fut in p_fwd_futures]
        w_evaluations_list = [fut.result() for fut in w_fwd_futures]
    
    print('Converted 32-column trace into polynomials and LDE extended them')

    # Compute constraint polynomials using degree-2 decomposition
    # Two constraint types:
    # C1_j: w_j(x) - (P_j(x) + rc_j(x))² = 0  (squaring constraint, degree 2)
    # C2_j: P_j(next) - MDS(sbox_state) = 0     (transition, degree 2 since sbox = w*(P+rc))
    c1_evaluations_list = [[] for _ in range(16)]  # squaring constraints
    c2_evaluations_list = [[] for _ in range(16)]  # transition constraints

    # === Optimized constraint evaluation loop ===
    # Cache globals as locals for faster access in hot loop
    _m = modulus
    _ef = extension_factor
    _prec = precision
    # Pre-extract MDS rows as tuples (avoids dict/list lookup per iteration)
    _mds = [tuple(MATRIX_FULL[j]) for j in range(16)]
    # Pre-extract column arrays to avoid nested indexing
    _p = p_evaluations_list
    _w = w_evaluations_list
    _rc = rc_evals
    _sf = s_full_evals
    # Pre-allocate append references
    _c1_append = [c1_evaluations_list[j].append for j in range(16)]
    _c2_append = [c2_evaluations_list[j].append for j in range(16)]

    for i in range(_prec):
        s_f = _sf[i]
        
        # Extract per-point column values once
        p_i = [_p[k][i] for k in range(16)]
        w_i = [_w[k][i] for k in range(16)]
        rc_i = [_rc[k][i] for k in range(16)]
        
        # state_plus_rc and C1 squaring constraint in one pass
        spr = [0] * 16
        for j in range(16):
            v = (p_i[j] + rc_i[j]) % _m
            spr[j] = v
            _c1_append[j]((w_i[j] - v * v % _m) % _m)
        
        # C2: transition constraint using w for S-box (degree 2)
        # sbox_state[0] = w[0] * spr[0]  (always, state[0] always gets S-box)
        sbox = [0] * 16
        sbox[0] = w_i[0] * spr[0] % _m
        for k in range(1, 16):
            val = spr[k]
            sbox[k] = (val + s_f * (w_i[k] * val % _m - val)) % _m
            
        # MDS matrix-vector multiply + transition check
        next_idx = (i + _ef) % _prec
        for j in range(16):
            r = _mds[j]
            mat_res = (
                r[0]*sbox[0] + r[1]*sbox[1] + r[2]*sbox[2] + r[3]*sbox[3] +
                r[4]*sbox[4] + r[5]*sbox[5] + r[6]*sbox[6] + r[7]*sbox[7] +
                r[8]*sbox[8] + r[9]*sbox[9] + r[10]*sbox[10] + r[11]*sbox[11] +
                r[12]*sbox[12] + r[13]*sbox[13] + r[14]*sbox[14] + r[15]*sbox[15]
            ) % _m
            _c2_append[j]((_p[j][next_idx] - mat_res) % _m)
        
    print('Computed 32 constraint polynomials (16 squaring + 16 transition)')

    # Vanishing polynomial components
    last_step_position = xs[(steps-1)*extension_factor]
    z_num_evaluations = [xs[(i * steps) % precision] - 1 for i in range(precision)]
    z_num_inv = f.multi_inv(z_num_evaluations)
    z_den_evaluations = [xs[i] - last_step_position for i in range(precision)]
    
    # Boundary constraint polynomials B_j
    b_evaluations_list = []
    valid_step_position = xs[valid_steps * extension_factor]
    zeropoly2 = f.mul_polys([-1, 1], [-valid_step_position, 1]) 
    inv_z2_evaluations = f.multi_inv([f.eval_poly_at(zeropoly2, x) for x in xs])
    
    for j in range(16):
        boundary_xs = [1, valid_step_position]
        boundary_ys = [input_state[j], output_state[j]]
        interpolant_j = f.lagrange_interp_2(boundary_xs, boundary_ys)
        i_j_evaluations = [f.eval_poly_at(interpolant_j, x) for x in xs]
        b_j_evals = [((p_evaluations_list[j][i] - i_j_evaluations[i]) * inv_z2_evaluations[i]) % modulus
                     for i in range(precision)]
        b_evaluations_list.append(b_j_evals)
    
    print('Computed 16 boundary constraint polynomials (B)')

    # Combine constraints with random weights
    k_c1 = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([50+j])), 'big') % modulus for j in range(16)]
    k_c2 = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    
    # D(x) = (Σ C1_j * k_c1[j] + Σ C2_j * k_c2[j]) / Z(x)
    d_evaluations = [0] * precision
    for i in range(precision):
        c_sum = (sum(c1_evaluations_list[j][i] * k_c1[j] for j in range(16)) +
                 sum(c2_evaluations_list[j][i] * k_c2[j] for j in range(16))) % modulus
        d_evaluations[i] = (c_sum * z_den_evaluations[i] * z_num_inv[i]) % modulus
    
    b_evaluations = [0] * precision
    for i in range(precision):
        b_evaluations[i] = sum(b_evaluations_list[j][i] * k_b[j] for j in range(16)) % modulus
    
    print('Computed D and B polynomials via weighted combination')
    
    # Build Merkle tree: 16 P + 16 W + B + D (34 field elements per leaf)
    mtree = merkelize([b''.join(p_evaluations_list[j][i].to_bytes(4, 'big') for j in range(16)) +
                       b''.join(w_evaluations_list[j][i].to_bytes(4, 'big') for j in range(16)) +
                       b_evaluations[i].to_bytes(4, 'big') +
                       d_evaluations[i].to_bytes(4, 'big')
                       for i in range(precision)])
    print('Computed hash root')

    # Random linear combination for FRI
    k1 = int.from_bytes(blake(mtree[1] + b'\x01'), 'big') % modulus
    k2 = int.from_bytes(blake(mtree[1] + b'\x02'), 'big') % modulus
    k3 = int.from_bytes(blake(mtree[1] + b'\x03'), 'big') % modulus
    k4 = int.from_bytes(blake(mtree[1] + b'\x04'), 'big') % modulus
    k_p = [int.from_bytes(blake(mtree[1] + bytes([10+j])), 'big') % modulus for j in range(16)]
    k_w = [int.from_bytes(blake(mtree[1] + bytes([30+j])), 'big') % modulus for j in range(16)]

    G2_to_the_steps = f.exp(G2, steps)
    powers = [1]
    for i in range(1, precision):
        powers.append(powers[-1] * G2_to_the_steps % modulus)

    l_evaluations = [0] * precision
    for i in range(precision):
        p_sum = sum(p_evaluations_list[j][i] * k_p[j] for j in range(16)) % modulus
        w_sum = sum(w_evaluations_list[j][i] * k_w[j] for j in range(16)) % modulus
        pw_sum = (p_sum + w_sum) % modulus
        pw_sum_weighted = (pw_sum * k1 + pw_sum * k2 * powers[i]) % modulus
        
        l_evaluations[i] = (d_evaluations[i] +
                            pw_sum_weighted +
                            b_evaluations[i] * k3 + b_evaluations[i] * powers[i] * k4) % modulus

    l_mtree = merkelize([val.to_bytes(4, 'big') for val in l_evaluations])
    print('Computed random linear combination')

    branches = []
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    print('Computed %d spot checks' % samples)

    # FRI degree bound: steps * 2 (down from steps * 3 thanks to degree-2 decomposition!)
    o = [mtree[1],
         l_mtree[1],
         mk_multi_branch(mtree, curr_next_positions),
         mk_multi_branch(l_mtree, positions),
         prove_low_degree(l_evaluations, G2, steps * 2, modulus, exclude_multiples_of=extension_factor)]
    print("Poseidon STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a Poseidon STARK
def verify_poseidon_proof(inp, steps, output, proof):
    import time
    from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2
    
    # OPTION 1: Transform steps to mixed-radix form (same as in mk_poseidon_proof)
    logsteps_input = steps.bit_length() - 1 if steps > 0 else 0
    logsteps_2radix = max(0, logsteps_input - 7)
    steps = (2 ** logsteps_2radix) * 127
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    # Since public polynomial we do them again here
    precision = steps * extension_factor
    G2 = f.exp(nonresidue, (modulus-1)//precision)
    skips = precision // steps
    G1 = f.exp(G2, skips)

    s_full_trace = []
    rc_traces = [[] for _ in range(16)]
    
    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        s_full_trace.append(is_full)
        if is_full:
            for j in range(16):
                rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                rc_idx += 1
        else:
            rc_traces[0].append(ROUND_CONSTANTS[rc_idx])
            rc_idx += 1
            for j in range(1, 16):
                rc_traces[j].append(0)
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    s_full_poly = fft(s_full_trace, modulus, G1, inv=True)
    rc_polys = [fft(rc_traces[j], modulus, G1, inv=True) for j in range(16)]
    
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    start_time = time.time()
    
    # Updated constraints for mixed-radix form
    assert steps <= 2**24 * 127, f"steps {steps} exceeds maximum domain"
    assert (modulus - 1) % steps == 0, f"steps {steps} must divide (p-1)"

    precision = steps * extension_factor

    # Get steps-th root of unity
    G2 = f.exp(nonresidue, (modulus-1)//precision)
    skips = precision // steps

    # First verify that the output matches the input by running the Poseidon computation directly
    computed_state = [inp % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rc_idx = 0
    for i in range(((steps - 1) // rounds_per_hash) * rounds_per_hash):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        if is_full:
            computed_state = [(computed_state[j] + ROUND_CONSTANTS[rc_idx + j]) % modulus for j in range(16)]
            computed_state = [pow(x, DEFAULT_ALPHA, modulus) for x in computed_state]
            rc_idx += 16
        else:
            computed_state[0] = (computed_state[0] + ROUND_CONSTANTS[rc_idx]) % modulus
            computed_state[0] = pow(computed_state[0], DEFAULT_ALPHA, modulus)
            rc_idx += 1
        computed_state = [sum(MATRIX_FULL[r][c] * computed_state[c] for c in range(16)) % modulus for r in range(16)]
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    assert computed_state[0] == output, "Poseidon output mismatch"
    print('Verified Poseidon computation')

    # Attempt FRI verification 
    verify_low_degree_proof(l_root, G2, fri_proof, steps * 2, modulus, exclude_multiples_of=extension_factor)
    print('FRI verification passed')

    # Performs the spot checks
    # Derive random coefficients deterministically from input and steps (same as prover)
    k_c1 = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([50+j])), 'big') % modulus for j in range(16)]
    k_c2 = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(4, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    
    # Coefficients for linear combination
    k1 = int.from_bytes(blake(m_root + b'\x01'), 'big') % modulus
    k2 = int.from_bytes(blake(m_root + b'\x02'), 'big') % modulus
    k3 = int.from_bytes(blake(m_root + b'\x03'), 'big') % modulus
    k4 = int.from_bytes(blake(m_root + b'\x04'), 'big') % modulus
    k_p = [int.from_bytes(blake(m_root + bytes([10+j])), 'big') % modulus for j in range(16)]
    k_w = [int.from_bytes(blake(m_root + bytes([30+j])), 'big') % modulus for j in range(16)]
    
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    last_step_position = f.exp(G2, (steps - 1) * skips)
    valid_steps = ((steps - 1) // rounds_per_hash) * rounds_per_hash
    valid_step_position = f.exp(G2, valid_steps * skips)
    main_branch_leaves = verify_multi_branch(m_root, curr_next_positions, main_branches)
    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)
    
    # Boundary polynomial Z2(x)
    zeropoly2 = f.mul_polys([-1, 1], [-valid_step_position, 1])

    for i, pos in enumerate(positions):
        x = f.exp(G2, pos)
        x_to_the_steps = f.exp(x, steps)
        mbranch1 = main_branch_leaves[i*2]
        mbranch2 = main_branch_leaves[i*2+1]
        l_of_x = int.from_bytes(linear_comb_branch_leaves[i], 'big')

        # Extract 16 P + 16 W + B + D from leaves (34 field elements, each 4 bytes)
        p_of_x = [int.from_bytes(mbranch1[j*4:(j+1)*4], 'big') for j in range(16)]
        p_of_next = [int.from_bytes(mbranch2[j*4:(j+1)*4], 'big') for j in range(16)]
        w_of_x = [int.from_bytes(mbranch1[(16+j)*4:(17+j)*4], 'big') for j in range(16)]
        b_of_x = int.from_bytes(mbranch1[32*4:33*4], 'big')
        d_of_x = int.from_bytes(mbranch1[33*4:34*4], 'big')

        zvalue = f.div(f.exp(x, steps) - 1, x - last_step_position)

        # Check transition constraints for all 16 columns
        s_f = f.eval_poly_at(s_full_poly, x)
        rcs = [f.eval_poly_at(rc_polys[k], x) for k in range(16)]
        
        state_plus_rc = [(p_of_x[k] + rcs[k]) % modulus for k in range(16)]
        
        # Verify squaring constraints: w[k] = (P[k] + rc[k])²
        c1_sum = 0
        for j in range(16):
            c1_j = (w_of_x[j] - pow(state_plus_rc[j], 2, modulus)) % modulus
            c1_sum = (c1_sum + c1_j * k_c1[j]) % modulus
        
        # Verify transition constraints using degree-2 decomposition
        # sbox_state[0] = w[0] * (P[0]+rc[0])  (always)
        # sbox_state[k] = (P[k]+rc[k]) + s_f * (w[k]*(P[k]+rc[k]) - (P[k]+rc[k]))
        sbox_state = [(w_of_x[0] * state_plus_rc[0]) % modulus]
        for k in range(1, 16):
            val = state_plus_rc[k]
            sbox_k = (val + s_f * ((w_of_x[k] * val) % modulus - val)) % modulus
            sbox_state.append(sbox_k)
        
        c2_sum = 0
        for j in range(16):
            mat_full_res = sum(MATRIX_FULL[j][k] * sbox_state[k] for k in range(16)) % modulus
            c2_j = (p_of_next[j] - mat_full_res) % modulus
            c2_sum = (c2_sum + c2_j * k_c2[j]) % modulus
        
        d_sum = (c1_sum + c2_sum) % modulus
        
        # Verify: sum of (P_j(g1*x) - P_j(x)) * k_c[j] = Z(x) * D(x)
        if (d_sum - zvalue * d_of_x) % modulus != 0:
            print(f'pos = {pos}')
            print(f'd_sum = {d_sum}')
            print(f'zvalue = {zvalue}')
            print(f'd_of_x = {d_of_x}')
            print(f'(zvalue * d_of_x) % mod = {(zvalue * d_of_x) % modulus}')
            print(f's_f = {s_f}')
            print(f'state_plus_rc = {state_plus_rc}')
            print(f'sbox_state = {sbox_state}')
            print(f'p_of_next = {p_of_next}')
        assert (d_sum - zvalue * d_of_x) % modulus == 0, f"Transition constraint failed at position {pos}"
        
        # Check boundary constraint
        b_sum_expected = 0
        z2_x = f.eval_poly_at(zeropoly2, x)
        for j in range(16):
            interpolant_j = f.lagrange_interp_2([1, valid_step_position], [inp % modulus if j == 0 else 0, computed_state[j]])
            i_j_x = f.eval_poly_at(interpolant_j, x)
            b_j_expected = (p_of_x[j] - i_j_x) % modulus
            b_sum_expected = (b_sum_expected + b_j_expected * k_b[j]) % modulus
            
        assert (b_sum_expected - b_of_x * z2_x) % modulus == 0, f"Boundary constraint failed at position {pos}"

        # Check correctness of the linear combination (now includes w columns)
        p_sum = sum(p_of_x[j] * k_p[j] for j in range(16)) % modulus
        w_sum = sum(w_of_x[j] * k_w[j] for j in range(16)) % modulus
        pw_sum = (p_sum + w_sum) % modulus
        pw_sum_weighted = (pw_sum * k1 + pw_sum * k2 * x_to_the_steps) % modulus
        
        expected_l = (d_of_x +
                      pw_sum_weighted +
                      b_of_x * k3 + b_of_x * x_to_the_steps * k4) % modulus
        
        assert (l_of_x - expected_l) % modulus == 0, f"Linear combination mismatch at position {pos}"

    print('Verified %d consistency checks' % spot_check_security_factor)
    print('Verified Poseidon STARK in %.4f sec' % (time.time() - start_time))
    return True