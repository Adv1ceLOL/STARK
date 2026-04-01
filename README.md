# STARK Proof System for Poseidon2

This repository contains a formal Python implementation of a Scalable Transparent ARguments of Knowledge (STARK) proof system tailored for the Poseidon2 hash function.

## Overview

The scheme constructs and verifies zero-knowledge proofs for the computational trace of Poseidon2. Operations are defined over the Koala prime field ($p = 2^{31} - 2^{24} + 1$). It achieves short proofs and fast verification through Fast Reed-Solomon Interactive Oracle Proofs of Proximity (FRI).

## Architecture

- **`poseidon2_stark.py`**: The core STARK prover and verifier. Handles computational traces, transition constraints, boundary constraints, and resulting Merkle commitments.
- **`poseidon2.py`**: A standalone reference implementation of the Poseidon2 cryptographic hash function.
- **`fri.py` / `fft.py`**: Implementations of the FRI testing protocol and Fast Fourier Transforms for low-degree polynomial evaluations.
- **`test.py`**: Assessment and integration test suites validating Merkle tree integrity, polynomial bounds, and STARK proof correctness.

## Usage

To execute the test suite generating and verifying a full Poseidon2 STARK proof, run:

```bash
python test.py [LOG_STEPS]
```

By default, the test suite generates a proof for $2^{13}$ trace steps.