from mixed_radix import fft
from poseidon_stark import mk_poseidon_proof, verify_poseidon_proof
from merkle_tree import merkelize, mk_branch, verify_branch, bin_length
from fri import prove_low_degree, verify_low_degree_proof

modulus = 2**24 *127 + 1


def test_merkletree():
    t = merkelize([x.to_bytes(32, 'big') for x in range(128)])
    b = mk_branch(t, 59)
    assert verify_branch(t[1], 59, b, output_as_int=True) == 59
    print('Merkle tree works')

def fri_proof_bin_length(fri_proof):
    return sum([32 + bin_length(x[1]) + bin_length(x[2]) for x in fri_proof[:-1]]) + len(b''.join(fri_proof[-1]))
    
def test_fri():
    # Pure FRI tests
    poly = list(range(4096))
    root_of_unity = pow(7, (modulus-1)//16384, modulus)
    evaluations = fft(poly, modulus, root_of_unity)
    proof = prove_low_degree(evaluations, root_of_unity, 4096, modulus)
    print("Approx proof length: %d" % fri_proof_bin_length(proof))
    assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 4096, modulus)
    
    try:
        fakedata = [x if pow(3, i, 4096) > 400 else 39 for x, i in enumerate(evaluations)]
        proof2 = prove_low_degree(fakedata, root_of_unity, 4096, modulus)
        assert verify_low_degree_proof(merkelize(fakedata)[1], root_of_unity, proof, 4096, modulus)
        raise Exception("Fake data passed FRI")
    except:
        pass
    try:
        assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 2048, modulus)
        raise Exception("Fake data passed FRI")
    except:
        pass

def test_stark():
    import sys
    LOGSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    # Full STARK test with Poseidon
    INPUT = 3
    proof = mk_poseidon_proof(INPUT, 2**LOGSTEPS)
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    L1 = bin_length(main_branches) + bin_length(linear_comb_branches)
    L2 = fri_proof_bin_length(fri_proof)
    print("Approx proof length: %d (branches), %d (FRI proof), %d (total)" % (L1, L2, L1 + L2))
    
    # Compute the output correctly by applying Poseidon
    from poseidon_stark import poseidon_full_state, DEFAULT_RF, DEFAULT_RP
    
    # Calculate actual steps used in STARK (mixed-radix form: 2^k × 127)
    logsteps_2radix = max(0, LOGSTEPS - 7)
    steps_actual = (2 ** logsteps_2radix) * 127
    
    # Adjust steps to be a multiple of rounds_per_hash for correct computation
    rounds_per_hash = DEFAULT_RF + DEFAULT_RP
    valid_steps = (steps_actual // rounds_per_hash) * rounds_per_hash
    
    final_state = poseidon_full_state(INPUT, valid_steps)
    output = final_state[0]
    print(f"Debug Output: {output}")
    assert verify_poseidon_proof(INPUT, 2**LOGSTEPS, output, proof)

if __name__ == '__main__':
    test_stark()
