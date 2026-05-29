"""
Debug / inspection script for fri.py functions.
Run this file to step through prove_low_degree and verify_low_degree_proof
with visible intermediate variables at every stage.
"""

from mixed_radix import fft
from poly_utils import PrimeField
from permuted_tree import merkelize, mk_branch, verify_branch
from utils import get_power_cycle, get_pseudorandom_indices
from fri import prove_low_degree, verify_low_degree_proof

# ── Constants ────────────────────────────────────────────────────────
modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
print(f"PrimeField instance f  : {f}")

# Evaluation domain size = 16384, polynomial degree < 4096
DOMAIN_SIZE = 16384
MAX_DEG_PLUS_1 = 4096
root_of_unity = pow(7, (modulus - 1) // DOMAIN_SIZE, modulus)

# ── Helper ───────────────────────────────────────────────────────────
def sep(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

# =====================================================================
#  1.  Setup: build a test polynomial and evaluate it
# =====================================================================
sep("1. Setup")

# Simple polynomial: coeffs = [0, 1, 2, ..., 4095]
poly_coeffs = list(range(MAX_DEG_PLUS_1))
print(f"Polynomial degree      : {len(poly_coeffs) - 1}")
print(f"Domain size            : {DOMAIN_SIZE}")
print(f"modulus                : {modulus}")
print(f"root_of_unity          : {root_of_unity}")

# Evaluate via FFT (pad coeffs to domain size with zeros)
padded_coeffs = poly_coeffs + [0] * (DOMAIN_SIZE - len(poly_coeffs))
evaluations = fft(padded_coeffs, modulus, root_of_unity)

print(f"Number of evaluations  : {len(evaluations)}")
print(f"First 5 evaluations    : {evaluations[:5]}")
print(f"Last  5 evaluations    : {evaluations[-5:]}")

# =====================================================================
#  2.  Merkle-tree of evaluations (used as the commitment root)
# =====================================================================
sep("2. Merkle commitment of evaluations")

m_tree = merkelize(evaluations)
merkle_root = m_tree[1]
print(f"Merkle root (hex)      : {merkle_root.hex()}")
print(f"Tree length            : {len(m_tree)}")

# =====================================================================
#  3.  Power cycle sanity check
# =====================================================================
sep("3. Power-cycle sanity check")

xs = get_power_cycle(root_of_unity, modulus)
print(f"Power cycle length     : {len(xs)}")
print(f"First 5 xs             : {xs[:5]}")
print(f"xs matches domain size : {len(xs) == DOMAIN_SIZE}")

# =====================================================================
#  4.  prove_low_degree  —  generate the FRI proof
# =====================================================================
sep("4. prove_low_degree")

proof = prove_low_degree(evaluations, root_of_unity, MAX_DEG_PLUS_1, modulus)

print(f"\nNumber of FRI rounds   : {len(proof)}")
for i, layer in enumerate(proof):
    if isinstance(layer, list) and len(layer) == 3:
        root2, col_branches, poly_branches = layer
        print(f"  Round {i}: root2={root2.hex()[:24]}...  "
              f"col_branches_len={len(col_branches)}  "
              f"poly_branches_len={len(poly_branches)}")
    else:
        # Final layer is raw values
        print(f"  Round {i} (final): {len(layer)} raw values")

# =====================================================================
#  5.  verify_low_degree_proof  —  verify the FRI proof
# =====================================================================
sep("5. verify_low_degree_proof")

try:
    ok = verify_low_degree_proof(
        merkle_root, root_of_unity, proof, MAX_DEG_PLUS_1, modulus
    )
    print(f"Verification result    : {ok}")
except AssertionError as e:
    print(f"Verification FAILED    : {e}")
except Exception as e:
    print(f"Verification ERROR     : {type(e).__name__}: {e}")

# =====================================================================
#  6.  Negative test: tamper with evaluations, expect failure
# =====================================================================
sep("6. Negative test – tampered evaluations")

bad_evaluations = list(evaluations)
# Flip some values
for i in range(0, len(bad_evaluations), 10):
    bad_evaluations[i] = (bad_evaluations[i] + 1) % modulus

bad_tree = merkelize(bad_evaluations)
bad_root = bad_tree[1]

try:
    bad_proof = prove_low_degree(bad_evaluations, root_of_unity, MAX_DEG_PLUS_1, modulus)
    result = verify_low_degree_proof(bad_root, root_of_unity, bad_proof, MAX_DEG_PLUS_1, modulus)
    print(f"Tampered verification  : {result}  (should have failed!)")
except AssertionError:
    print("Tampered proof correctly REJECTED (AssertionError)")
except Exception as e:
    print(f"Tampered proof correctly REJECTED: {type(e).__name__}: {e}")

# =====================================================================
#  7.  Negative test: wrong maxdeg_plus_1
# =====================================================================
sep("7. Negative test – wrong max degree claim")

try:
    # Prove degree < 4096 but verify claiming degree < 2048
    result = verify_low_degree_proof(
        merkle_root, root_of_unity, proof, MAX_DEG_PLUS_1 // 2, modulus
    )
    print(f"Wrong-degree verify    : {result}  (should have failed!)")
except AssertionError:
    print("Wrong degree correctly REJECTED (AssertionError)")
except Exception as e:
    print(f"Wrong degree correctly REJECTED: {type(e).__name__}: {e}")

# =====================================================================
#  8.  Small polynomial test (hits the base-case branch)
# =====================================================================
sep("8. Small polynomial (base-case, degree <= 15)")

SMALL_DOMAIN = 64
SMALL_DEG = 16
small_rou = pow(7, (modulus - 1) // SMALL_DOMAIN, modulus)
small_poly = list(range(SMALL_DEG))
small_padded = small_poly + [0] * (SMALL_DOMAIN - SMALL_DEG)
small_evals = fft(small_padded, modulus, small_rou)

print(f"Small domain size      : {SMALL_DOMAIN}")
print(f"Small maxdeg+1         : {SMALL_DEG}")

small_proof = prove_low_degree(small_evals, small_rou, SMALL_DEG, modulus)
print(f"Small proof rounds     : {len(small_proof)}")

small_tree = merkelize(small_evals)
try:
    ok = verify_low_degree_proof(small_tree[1], small_rou, small_proof, SMALL_DEG, modulus)
    print(f"Small verify result    : {ok}")
except Exception as e:
    print(f"Small verify ERROR     : {type(e).__name__}: {e}")

# =====================================================================
#  9.  Inspect PrimeField helpers used inside FRI
# =====================================================================
sep("9. PrimeField helpers spot-check")

# multi_interp_4 & eval_quartic are the core of each FRI round
quarter = len(xs) // 4
sample_xs = [[xs[i + quarter * j] for j in range(4)] for i in range(4)]
sample_ys = [[evaluations[i + quarter * j] for j in range(4)] for i in range(4)]

polys_4 = f.multi_interp_4(sample_xs, sample_ys)
print(f"multi_interp_4 output (first 4 quartic polys):")
for idx, p in enumerate(polys_4):
    print(f"  poly[{idx}] = {[hex(c)[:18]+'...' for c in p]}")
    # Quick check: re-evaluate at the original x-points
    for j in range(4):
        val = f.eval_quartic(p, sample_xs[idx][j])
        assert val == sample_ys[idx][j], \
            f"eval_quartic mismatch at row {idx}, col {j}"
print("  eval_quartic round-trip: OK")

# =====================================================================
sep("All done!")
