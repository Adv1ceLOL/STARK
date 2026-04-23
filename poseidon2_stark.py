from permuted_tree import merkelize, mk_branch, verify_branch, blake, mk_multi_branch, verify_multi_branch
from poly_utils import PrimeField
import time
from fft import fft
from fri import prove_low_degree, verify_low_degree_proof
from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2
from poseidon2 import poseidon2, ROUND_CONSTANTS, MATRIX_FULL, MATRIX_PARTIAL, DEFAULT_RF, DEFAULT_RP, DEFAULT_ALPHA, P, mat_vec_mul

# Single modulus for all operations (Koala modulus)
modulus = 2**31 - 2**24 + 1

f = PrimeField(modulus)
nonresidue = 3  # Better primitive root for this modulus than 7

spot_check_security_factor = 80
extension_factor = 4  # Now works with base 3 as primitive root

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
def mk_poseidon2_proof(inp, steps):
    start_time = time.time()
    # Some constraints to make our job easier
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps)

    # Total Size of the evaluation domain
    precision = steps * extension_factor

    # Root of unity such that x^precision=1
    G2 = f.exp(nonresidue, (modulus-1)//precision)

    # Root of unity such that x^steps=1
    skips = precision // steps
    G1 = f.exp(G2, skips)

    # Powers of the higher-order root of unity
    xs = get_power_cycle(G2, modulus)
    last_step_position = xs[(steps-1)*extension_factor]

    # Generate public polynomials for selectors and round constants
    s_full_trace = []
    rc_traces = [[] for _ in range(16)]
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    rc_idx = 0
    for i in range(steps):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        s_full_trace.append(is_full)
        for j in range(16):
            if is_full:
                rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                rc_idx += 1
            else:
                if j == 0:
                    rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                    rc_idx += 1
                else:
                    rc_traces[j].append(0)
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    s_full_poly = fft(s_full_trace, modulus, G1, inv=True)
    s_full_evals = fft(s_full_poly, modulus, G2)
    
    rc_polys = [fft(rc_traces[j], modulus, G1, inv=True) for j in range(16)]
    rc_evals = [fft(rc_polys[j], modulus, G2) for j in range(16)]
    
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
        
        current_state = [(current_state[j] + rc_traces[j][i]) % modulus for j in range(16)]
        if is_full:
            current_state = [pow(x, DEFAULT_ALPHA, modulus) for x in current_state]
            current_state = mat_vec_mul(MATRIX_FULL, current_state, modulus)
        else:
            current_state[0] = pow(current_state[0], DEFAULT_ALPHA, modulus)
            current_state = mat_vec_mul(MATRIX_PARTIAL, current_state, modulus)
            
        computational_trace.append(list(current_state))
    
    # Extract input and output states
    input_state = computational_trace[0]
    output_state = computational_trace[-1]
    print('Done generating poseidon computational trace (16-column)')

    # Interpolate each of the 16 state element columns into a polynomial
    # P_j(x) is the polynomial whose evaluations at powers of G1 are the j-th state element across steps
    p_polynomials = []
    p_evaluations_list = []
    
    for state_idx in range(16):
        # Extract column j from the trace
        trace_column = [computational_trace[i][state_idx] for i in range(steps)]
        
        # Interpolate into polynomial
        p_poly = fft(trace_column, modulus, G1, inv=True)
        p_polynomials.append(p_poly)
        
        # Low-degree extend over the larger domain G2
        p_evals = fft(p_poly, modulus, G2)
        p_evaluations_list.append(p_evals)
    
    print('Converted 16-column computational trace into polynomials and low-degree extended them')

    # Compute the 16 composed polynomials C_j
    c_evaluations_list = []

    for j in range(16):
        c_j_evals = []
        for i in range(precision):
            s_f = s_full_evals[i]
            
            state_plus_rc = [(p_evaluations_list[k][i] + rc_evals[k][i]) % modulus for k in range(16)]
            sbox_state = [pow(state_plus_rc[0], DEFAULT_ALPHA, modulus)] + [
                (pow(state_plus_rc[k], DEFAULT_ALPHA, modulus) * s_f + state_plus_rc[k] * (1 - s_f)) % modulus
                for k in range(1, 16)
            ]
            
            mat_full_res = sum(MATRIX_FULL[j][k] * sbox_state[k] for k in range(16)) % modulus
            mat_partial_res = sum(MATRIX_PARTIAL[j][k] * sbox_state[k] for k in range(16)) % modulus
            
            p_next_calculated = (mat_full_res * s_f + mat_partial_res * (1 - s_f)) % modulus
            
            next_p_j = p_evaluations_list[j][(i + extension_factor) % precision]
            c_j_evals.append((next_p_j - p_next_calculated) % modulus)
        c_evaluations_list.append(c_j_evals)
        
    print('Computed 16 transition constraint polynomials')

    # Compute D(x) = (sum of C_j(x) weighted by random coefficients) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    # First, generate random coefficients for combining the 16 C polynomials
    # We'll compute this after we have a Merkle root to derive randomness
    # For now, compute the basis: numerator of Z(x) and its inverses
    
    last_step_position = xs[(steps-1)*extension_factor]
    z_num_evaluations = [xs[(i * steps) % precision] - 1 for i in range(precision)]
    z_num_inv = f.multi_inv(z_num_evaluations)
    z_den_evaluations = [xs[i] - last_step_position for i in range(precision)]
    
    # Compute the B polynomial using all 16 state elements at boundaries
    # For each state element j, we need: B_j(x) * Q(x) + I_j(x) = P_j(x)
    # where I_j interpolates input_state[j] at x=1 and output_state[j] at x=x_atlast_step
    
    b_evaluations_list = []
    #(x−1)(x−last_step_position)
    zeropoly2 = f.mul_polys([-1, 1], [-last_step_position, 1]) 
    # 1 / (Q(G2^i)) for all i
    inv_z2_evaluations = f.multi_inv([f.eval_poly_at(zeropoly2, x) for x in xs])
    
    for j in range(16):
        # Interpolate boundaries for state element j in 2 Points
        # Point 1:(x,y)= (1, input_state[j])
        # Point 2:(x,y)= (last_step_position, output_state[j])
        boundary_xs = [1, last_step_position]
        boundary_ys = [input_state[j], output_state[j]]
        # interpolant_j  = [c0,c1] (degree 1 Polynomial I_j(x))
        interpolant_j = f.lagrange_interp_2(boundary_xs, boundary_ys)
        # Evaluates I_j(x) at all precision points
        i_j_evaluations = [f.eval_poly_at(interpolant_j, x) for x in xs]
        # Compute B_j(x) = (P_j(x) - I_j(x)) / Q(x)
        b_j_evals = [((p_evaluations_list[j][i] - i_j_evaluations[i]) * inv_z2_evaluations[i]) % modulus
                     for i in range(precision)]
        b_evaluations_list.append(b_j_evals)
    
    print('Computed 16 boundary constraint polynomials (B)')

    # Compute D and B via weighted combination of their 16 components
    # Derive random coefficients from input value and steps (deterministic for prover and verifier)
    # Random Weights for each B_j and C_j
    k_c = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    
    # Compute D(x) = (Σ C_j(x) * k_c[j]) / Z(x) at all evaluation points
    # Z(x) = (x^steps - 1) / (x - last_step_position)
    # At each evaluation point x_i = G2^i:
    #   D(x_i) = (Σ C_j(x_i) * k_c[j]) * (x_i - last) / ((x_i)^steps - 1)
    d_evaluations = [0] * precision
    for i in range(precision):
        # Numerator
        c_sum = sum(c_evaluations_list[j][i] * k_c[j] for j in range(16)) % modulus
        # Denominator: divide by Z(x) = (x^steps - 1) / (x - last_step_position)
        # Computing as: c_sum * (x - last) / (x^steps - 1)
        d_evaluations[i] = (c_sum * z_den_evaluations[i] * z_num_inv[i]) % modulus
    
    # Compute B_combined(x) = sum of B_j(x) * k_b[j]
    b_evaluations = [0] * precision
    for i in range(precision):
        b_evaluations[i] = sum(b_evaluations_list[j][i] * k_b[j] for j in range(16)) % modulus
    
    print('Computed D and B polynomials via weighted combination')
    
    # Now build the final Merkle tree with all components: 16 P + B + D
    mtree = merkelize([b''.join(p_evaluations_list[j][i].to_bytes(32, 'big') for j in range(16)) +
                       b_evaluations[i].to_bytes(32, 'big') +
                       d_evaluations[i].to_bytes(32, 'big')
                       for i in range(precision)])
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
    G2_to_the_steps = f.exp(G2, steps)
    powers = [1]
    for i in range(1, precision):
        powers.append(powers[-1] * G2_to_the_steps % modulus)

    l_evaluations = [0] * precision
    for i in range(precision):
        # Add weighted sum of all 16 P columns
        p_sum = sum(p_evaluations_list[j][i] * k_p[j] for j in range(16)) % modulus
        p_sum_weighted = (p_sum * k1 + p_sum * k2 * powers[i]) % modulus
        
        # Add B and D terms
        l_evaluations[i] = (d_evaluations[i] +
                            p_sum_weighted +
                            b_evaluations[i] * k3 + b_evaluations[i] * powers[i] * k4) % modulus

    l_mtree = merkelize([val.to_bytes(32, 'big') for val in l_evaluations])
    print('Computed random linear combination')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates, excluding
    # multiples of `extension_factor`
    branches = []
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [mtree[1],
         l_mtree[1],
         mk_multi_branch(mtree, curr_next_positions),
         mk_multi_branch(l_mtree, positions),
         prove_low_degree(l_evaluations, G2, steps * 4, modulus, exclude_multiples_of=extension_factor)]
    print("Poseidon2 STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a Poseidon2 STARK
def verify_poseidon2_proof(inp, steps, output, proof):
    import time
    from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2
    
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    half_rf = DEFAULT_RF // 2
    
    # Since public polynomial we do tehm again here
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
        for j in range(16):
            if is_full:
                rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                rc_idx += 1
            else:
                if j == 0:
                    rc_traces[j].append(ROUND_CONSTANTS[rc_idx])
                    rc_idx += 1
                else:
                    rc_traces[j].append(0)
        
        # Reset rc_idx for the next hash if we wrapped around
        if round_idx == rounds_per_hash - 1:
            rc_idx = 0

    s_full_poly = fft(s_full_trace, modulus, G1, inv=True)
    rc_polys = [fft(rc_traces[j], modulus, G1, inv=True) for j in range(16)]
    
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    start_time = time.time()
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps)

    precision = steps * extension_factor

    # Get steps-th root of unity
    G2 = f.exp(nonresidue, (modulus-1)//precision)
    skips = precision // steps

    # First verify that the output matches the actual Poseidon2 computation
    # For 16-column STARK, we verify the full state
    computed_state = [inp % modulus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(steps - 1):
        round_idx = i % rounds_per_hash
        is_full = 1 if (round_idx < half_rf or round_idx >= half_rf + DEFAULT_RP) else 0
        
        computed_state = [(computed_state[j] + rc_traces[j][i]) % modulus for j in range(16)]
        if is_full:
            computed_state = [pow(x, DEFAULT_ALPHA, modulus) for x in computed_state]
            computed_state = mat_vec_mul(MATRIX_FULL, computed_state, modulus)
        else:
            computed_state[0] = pow(computed_state[0], DEFAULT_ALPHA, modulus)
            computed_state = mat_vec_mul(MATRIX_PARTIAL, computed_state, modulus)

    assert computed_state[0] == output, "Poseidon2 output mismatch"
    print('Verified Poseidon2 computation')

    # Attempt FRI verification 
    verify_low_degree_proof(l_root, G2, fri_proof, steps * 4, modulus, exclude_multiples_of=extension_factor)
    print('FRI verification passed')

    # Performs the spot checks
    # Derive random coefficients deterministically from input and steps (same as prover)
    k_c = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([100+j])), 'big') % modulus for j in range(16)]
    k_b = [int.from_bytes(blake(inp.to_bytes(32, 'big') + steps.to_bytes(8, 'big') + bytes([200+j])), 'big') % modulus for j in range(16)]
    
    # Coefficients for linear combination
    k1 = int.from_bytes(blake(m_root + b'\x01'), 'big') % modulus
    k2 = int.from_bytes(blake(m_root + b'\x02'), 'big') % modulus
    k3 = int.from_bytes(blake(m_root + b'\x03'), 'big') % modulus
    k4 = int.from_bytes(blake(m_root + b'\x04'), 'big') % modulus
    k_p = [int.from_bytes(blake(m_root + bytes([10+j])), 'big') % modulus for j in range(16)]
    
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    curr_next_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    last_step_position = f.exp(G2, (steps - 1) * skips)
    main_branch_leaves = verify_multi_branch(m_root, curr_next_positions, main_branches)
    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)
    
    for i, pos in enumerate(positions):
        x = f.exp(G2, pos)
        x_to_the_steps = f.exp(x, steps)
        mbranch1 = main_branch_leaves[i*2]
        mbranch2 = main_branch_leaves[i*2+1]
        l_of_x = int.from_bytes(linear_comb_branch_leaves[i], 'big')

        # Extract the 16 P values, B, and D from the leaves
        # Leaf format: P_0 || P_1 || ... || P_15 || B || D (each 32 bytes)
        p_of_x = [int.from_bytes(mbranch1[j*32:(j+1)*32], 'big') for j in range(16)]
        p_of_next = [int.from_bytes(mbranch2[j*32:(j+1)*32], 'big') for j in range(16)]
        b_of_x = int.from_bytes(mbranch1[16*32:17*32], 'big')
        d_of_x = int.from_bytes(mbranch1[17*32:18*32], 'big')

        zvalue = f.div(f.exp(x, steps) - 1,
                       x - last_step_position)

        # Check transition constraints for all 16 columns
        s_f = f.eval_poly_at(s_full_poly, x)
        rcs = [f.eval_poly_at(rc_polys[k], x) for k in range(16)]
        
        state_plus_rc = [(p_of_x[k] + rcs[k]) % modulus for k in range(16)]
        sbox_state = [pow(state_plus_rc[0], DEFAULT_ALPHA, modulus)] + [(pow(state_plus_rc[k], DEFAULT_ALPHA, modulus) * s_f + state_plus_rc[k] * (1 - s_f)) % modulus for k in range(1, 16)]
        
        mat_full_res = sum(MATRIX_FULL[0][k] * sbox_state[k] for k in range(16)) % modulus
        mat_partial_res = sum(MATRIX_PARTIAL[0][k] * sbox_state[k] for k in range(16)) % modulus
        
        d_sum = 0

        for j in range(16):
            p_next_calculated = (mat_full_res * s_f + mat_partial_res * (1 - s_f)) % modulus
            c_j_contribution = (p_of_next[j] - p_next_calculated) % modulus
            d_sum = (d_sum + c_j_contribution * k_c[j]) % modulus
            
            if j < 15:
                mat_full_res = sum(MATRIX_FULL[j+1][k] * sbox_state[k] for k in range(16)) % modulus
                mat_partial_res = sum(MATRIX_PARTIAL[j+1][k] * sbox_state[k] for k in range(16)) % modulus
        
        # Verify: sum of (P_j(g1*x) - P_j(x)) * k_c[j] = Z(x) * D(x)
        assert (d_sum - zvalue * d_of_x) % modulus == 0, f"Transition constraint failed at position {pos}"

        # Check correctness of the linear combination
        p_sum = sum(p_of_x[j] * k_p[j] for j in range(16)) % modulus
        p_sum_weighted = (p_sum * k1 + p_sum * k2 * x_to_the_steps) % modulus
        
        expected_l = (d_of_x +
                      p_sum_weighted +
                      b_of_x * k3 + b_of_x * x_to_the_steps * k4) % modulus
        
        assert (l_of_x - expected_l) % modulus == 0, f"Linear combination mismatch at position {pos}"

    print('Verified %d consistency checks' % spot_check_security_factor)
    print('Verified Poseidon2 STARK in %.4f sec' % (time.time() - start_time))
    return True
