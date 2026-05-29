"""
Mixed-Radix FFT and Bluestain NTT implementations for Bluestain STARK.

This module provides efficient FFT algorithms for arbitrary field sizes:
- Cooley-Tukey 2-radix FFT (power-of-2 sizes)
- Mixed-radix NTT (composite sizes)
- Bluestain NTT (arbitrary sizes)

All operations use modular arithmetic over a prime field.
"""

def transpose(data, rows):
    """
    Transpose a matrix represented as a flat list.
    
    Given data as a flat array of length N = rows * cols,
    with layout: [row0_col0, row0_col1, ..., row0_colN, row1_col0, ...]
    Transforms to: [col0_row0, col1_row0, ..., colN_row0, col0_row1, ...]
    
    Args:
        data: List of field elements (mutable)
        rows: Number of rows in the original matrix
    
    Returns:
        Transposed data (modifies in-place)
    """
    n = len(data)
    if n == 0:
        return data
    
    cols = n // rows
    
    # Nothing to do if single row or single column
    if cols == 1 or rows == 1:
        return data
    
    # Perform transpose via temporary buffer
    tmp = [0] * n
    for m in range(n):
        j = m // cols  # Original row
        r = m - j * cols  # Original column
        new_idx = r * rows + j
        tmp[new_idx] = data[m]
    
    # Copy back
    for i in range(n):
        data[i] = tmp[i]
    
    return data


def get_primitive_root(p, order):
    """
    Find a primitive root of the given order in Z_p.
    
    Args:
        p: Prime modulus
        order: Desired multiplicative order
    
    Returns:
        A primitive root, or None if not found
    """
    # For efficiency, we use trial: check small numbers
    for candidate in range(2, min(p, 1000)):
        if pow(candidate, order, p) == 1:
            # Check if this is actually primitive (no smaller power gives 1)
            for d in range(1, order):
                if pow(candidate, d, p) == 1:
                    break
            else:
                return candidate
    return None


def get_omega(p, n, inverse=False):
    """
    Get a primitive n-th root of unity in Z_p.
    
    IMPORTANT MATHEMATICAL CONSTRAINT:
    For an n-th root of unity to exist, n must divide (p-1).
    This is a theorem in group theory, not a code limitation.
    
    For modulus p = 2^24 × 127 - 1:
    - (p-1) = 2 × (odd number)
    - Only n=1 and n=2 divide (p-1)
    - FFT is only possible for these sizes
    
    Args:
        p: Prime modulus
        n: Order of the root (FFT size)
        inverse: Ignored in this implementation
    
    Returns:
        A primitive n-th root of unity mod p
        
    Raises:
        ValueError: if n does not divide (p-1)
    """
    # Special cases
    if n == 1:
        return 1
    if n == 2:
        return p - 1  # -1 mod p (always a 2nd root for prime p)
    
    # General case: n must divide (p-1)
    if (p - 1) % n != 0:
        # Mathematical constraint: n must divide (p-1)
        # This is not a limitation of our code—it's group theory
        valid_sizes = []
        for test_n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if (p - 1) % test_n == 0:
                valid_sizes.append(test_n)
        raise ValueError(
            f"\n{'='*70}\n"
            f"FFT SIZE NOT SUPPORTED\n"
            f"{'='*70}\n"
            f"Cannot compute {n}-th root of unity mod p\n"
            f"Reason: {n} does not divide (p-1) = {p-1}\n"
            f"\n"
            f"This is a MATHEMATICAL THEOREM, not a code limitation:\n"
            f"  • For n-th roots to exist in Z_p: n | (p-1) (group theory)\n"
            f"  • (p-1) = 2 × {(p-1)//2} (only divisible by 2 and odd factors)\n"
            f"\n"
            f"Valid FFT sizes for this modulus: {valid_sizes}\n"
            f"\n"
            f"To use larger FFT sizes, switch to a modulus where\n"
            f"(p-1) has larger 2-power factors (e.g., Goldilocks Prime).\n"
            f"{'='*70}"
        )
    
    # Standard method: find primitive n-th root
    for candidate in range(2, min(p, 10000)):
        power_mod = pow(candidate, n, p)
        if power_mod == 1:
            # Verify it's primitive: check that candidate^(n/q) ≠ 1 for prime divisors q of n
            is_primitive = True
            
            # Check small prime divisors
            temp_n = n
            checked = set()
            for q in [2, 3, 5, 7, 11, 13]:
                if temp_n % q == 0 and q not in checked:
                    checked.add(q)
                    if pow(candidate, n // q, p) == 1:
                        is_primitive = False
                        break
                while temp_n % q == 0:
                    temp_n //= q
            
            if is_primitive and temp_n > 1:
                # temp_n is a larger prime factor, check it
                if pow(candidate, n // temp_n, p) == 1:
                    is_primitive = False
            
            if is_primitive:
                return candidate
    
    # Fallback: try mathematical construction
    for g in range(2, 1000):
        # Check if g is a primitive root of p
        if pow(g, (p - 1) // 2, p) != 1:  # Likely primitive
            omega = pow(g, (p - 1) // n, p)
            if pow(omega, n, p) == 1:
                return omega
    
    raise ValueError(
        f"Could not find primitive {n}-th root of unity mod {p}.\n"
        f"(This should not happen if n divides p-1)"
    )


def cooley_tukey_fft(values, modulus, omega, inverse=False):
    """
    Cooley-Tukey 2-radix FFT (iterative, in-place).
    
    Computes FFT of the given values using the Cooley-Tukey algorithm.
    Operates on power-of-2 sized inputs.
    
    Args:
        values: List of field elements (must be power of 2 length)
        modulus: Prime field modulus
        omega: Primitive n-th root of unity (where n = len(values))
        inverse: If True, compute inverse FFT
    
    Returns:
        Transformed values (modifies in-place)
    """
    n = len(values)
    
    # Validate input size
    if n == 0:
        return values
    if n == 1:
        return values
    
    # Check power of 2
    if (n & (n - 1)) != 0:
        raise ValueError(f"FFT size must be power of 2, got {n}")
    
    # Bit-reversal permutation
    j = 0
    for i in range(n - 1):
        if i < j:
            values[i], values[j] = values[j], values[i]
        
        k = n >> 1
        while k <= j:
            j -= k
            k >>= 1
        j += k
    
    # Precompute twiddle factors
    w_base = omega
    if inverse:
        w_base = pow(w_base, modulus - 2, modulus)
        
    twiddles = [1] * (n // 2)
    _mod = modulus  # local ref for hot loop
    for i in range(1, n // 2):
        twiddles[i] = twiddles[i-1] * w_base % _mod
        
    # Cooley-Tukey FFT computation (optimized inner loop)
    length = 2
    while length <= n:
        half = length >> 1
        step = n // length
        for i in range(0, n, length):
            for j in range(half):
                idx_lo = i + j
                idx_hi = idx_lo + half
                t = values[idx_hi] * twiddles[j * step] % _mod
                values[idx_hi] = (values[idx_lo] - t) % _mod
                values[idx_lo] = (values[idx_lo] + t) % _mod
        
        length <<= 1
    
    # For inverse FFT, scale by 1/n
    if inverse:
        n_inv = pow(n, _mod - 2, _mod)  # n^-1 mod modulus (Fermat)
        for i in range(n):
            values[i] = values[i] * n_inv % _mod
    
    return values


def cooley_tukey_fft_recursive(values, modulus, omega, inverse=False, n=None, start=0, stride=1):
    """
    Cooley-Tukey 2-radix FFT (recursive implementation for reference).
    
    Args:
        values: List of field elements
        modulus: Prime field modulus
        omega: Primitive n-th root of unity
        inverse: If True, compute inverse FFT
        n: Size of current FFT (defaults to len(values))
        start: Starting index in values array
        stride: Stride through values array
    
    Returns:
        List of FFT values
    """
    if n is None:
        n = len(values)
    
    if n <= 1:
        return [values[start]]
    
    if inverse:
        # For inverse: use ω^(-1)
        w = pow(omega, modulus - 2, modulus)  # Fermat inversion
        w_n = pow(w, (len(values) // n), modulus)
    else:
        w_n = pow(omega, (len(values) // n), modulus)
    
    # Split into even and odd
    even = cooley_tukey_fft_recursive(values, modulus, omega, inverse, n // 2, start, stride * 2)
    odd = cooley_tukey_fft_recursive(values, modulus, omega, inverse, n // 2, start + stride, stride * 2)
    
    # Combine
    result = [0] * n
    w = 1
    for k in range(n // 2):
        t = (odd[k] * w) % modulus
        result[k] = (even[k] + t) % modulus
        result[k + n // 2] = (even[k] - t) % modulus
        w = (w * w_n) % modulus
    
    if inverse and n == len(values):
        n_inv = pow(n, modulus - 2, modulus)
        for i in range(len(result)):
            result[i] = (result[i] * n_inv) % modulus
    
    return result


def fft_iterative(vals, modulus, omega, inverse=False):
    """
    Iterative 2-radix FFT wrapper.
    
    Args:
        vals: List of values to transform
        modulus: Prime field modulus
        omega: Primitive root of unity
        inverse: If True, compute inverse FFT
    
    Returns:
        Transformed values
    """
    # Make a copy to avoid modifying input
    values = list(vals)
    cooley_tukey_fft(values, modulus, omega, inverse)
    return values


def expand_root_of_unity(root_of_unity, modulus):
    """
    Build up all powers of a root of unity until we loop back to 1.
    
    Args:
        root_of_unity: The base root ω
        modulus: Prime modulus
    
    Returns:
        List [1, ω, ω², ..., ω^(order-1)] where ω^order ≡ 1
    """
    rootz = [1, root_of_unity]
    # Safety limit: prevent infinite loops for large orders
    # For 2^24 × 127 - 1 modulus, only valid orders are 1 and 2
    # But expand_root_of_unity may compute large roots, so limit to 2^24
    max_order = 2**24
    
    # NOTE: We do NOT check "if next_root in rootz" because:
    # 1. It's O(n) per iteration, making total O(n²) complexity
    # 2. For proper roots of unity, cycle only happens when rootz[-1] == 1
    # 3. We trust the root is correct, so we just check for 1 or max_order
    
    while rootz[-1] != 1 and len(rootz) < max_order:
        next_root = (rootz[-1] * root_of_unity) % modulus
        rootz.append(next_root)
    
    return rootz


def pad_to_power_of_2(n, min_size=1):
    """
    Find the smallest power of 2 that is >= n.
    
    Args:
        n: Desired minimum size
        min_size: Absolute minimum (default 1)
    
    Returns:
        Smallest power of 2 >= max(n, min_size)
    """
    size = max(n, min_size)
    power = 1
    while power < size:
        power *= 2
    return power


def tonelli_shanks_sqrt(a, p):
    """
    Compute square root of a modulo p using Tonelli-Shanks algorithm.
    Returns r such that r^2 ≡ a (mod p), or None if no square root exists.
    """
    # Quick check if a is a quadratic residue
    if pow(a, (p - 1) // 2, p) != 1:
        return None  # Not a quadratic residue
    
    # Factor p-1 as 2^s * q where q is odd
    s = 0
    q = p - 1
    while q % 2 == 0:
        s += 1
        q //= 2
    
    # If s == 1, then p ≡ 3 (mod 4), use simple formula
    if s == 1:
        return pow(a, (p + 1) // 4, p)
    
    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    # Tonelli-Shanks iteration
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)
    
    while True:
        if t == 0:
            return 0
        if t == 1:
            return r
        
        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Update values
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p


def bluestein_ntt(scalars, modulus, omega, inverse=False):
    """
    Bluestain NTT for arbitrary sizes.
    
    Implements the Bluestain algorithm to compute NTT for any size N,
    by reducing to a convolution of size O(N) which is then computed
    using 2-radix FFT on a power-of-2 padded size.
    
    Algorithm:
    1. Compute g = [ω^(j²/2) for j in 0..n]
    2. Rescale: x'_j = x_j * g_j
    3. Compute convolution of x' with g_inv = [ω^(-j²/2)]
    4. via FFT convolution on padded power-of-2 size
    5. Rescale back: result_j = convolution[j] * g_j
    
    Args:
        scalars: List of field elements (will be modified)
        modulus: Prime field modulus
        omega: Primitive n-th root of unity (where n = len(scalars))
        inverse: If True, compute inverse NTT
    
    Returns:
        Transformed values (modifies scalars in-place)
    """
    n = len(scalars)
    
    if n <= 1:
        return scalars
        
    if inverse:
        omega = pow(omega, modulus - 2, modulus)
        
    omega_sqrt = tonelli_shanks_sqrt(omega, modulus)
    if omega_sqrt is None:
        raise ValueError(f"omega ({omega}) has no square root modulo {modulus}")
    
    omega_inv = pow(omega, modulus - 2, modulus)  # Fermat inverse
    omega_inv_sqrt = pow(omega_sqrt, modulus - 2, modulus)
    
    # Generate g = [ω^(j²/2) for j in 0..n-1]
    g = [1] * n
    for j in range(1, n):
        jsquare = (j * j) % (2 * n)
        # ω^(j²/2) = (ω^(1/2))^(j²)
        g[j] = pow(omega_sqrt, jsquare, modulus)
    
    # Generate g_inv = [ω^(-j²/2) for j in 0..n-1]
    g_inv = [1] * n
    for j in range(1, n):
        jsquare = (j * j) % (2 * n)
        # ω^(-j²/2) = (ω^(-1/2))^(j²)
        g_inv[j] = pow(omega_inv_sqrt, jsquare, modulus)
    
    # Rescale input: x'_j = x_j * g_j
    for j in range(n):
        scalars[j] = (scalars[j] * g[j]) % modulus
    
    # Prepare for convolution: need to pad to power of 2
    padded_size = pad_to_power_of_2(2 * n - 1)
    
    # Build convolution kernel: [g_inv[0], g_inv[1], ..., g_inv[n-1], 0, ..., 0, g_inv[n-2], ..., g_inv[1]]
    # This is for cyclic convolution via FFT
    g_inv_padded = [0] * padded_size
    
    # Copy first n elements
    for j in range(n):
        g_inv_padded[j] = g_inv[j]
    
    # Copy reverse of elements 1..n-1 at the end
    for j in range(1, n):
        g_inv_padded[padded_size - n + j] = g_inv[n - j]
    
    # Pad scalars
    scalars_padded = scalars + [0] * (padded_size - n)
    
    # FFT both sequences
    get_padded_omega = pow(omega, n // padded_size, modulus) if n < padded_size else omega
    # Actually, we need the padded_size-th root
    # omega is n-th root, we need padded_size-th root
    # Use omega^(gcd(n, padded_size) / padded_size)? No...
    # We need a new root for the padded size
    
    # Simpler: compute a padded_size-th root of unity
    padded_omega = get_omega(modulus, padded_size, inverse=False)
    
    # Forward FFT
    scalars_padded = fft_iterative(scalars_padded, modulus, padded_omega, inverse=False)
    g_inv_padded = fft_iterative(g_inv_padded, modulus, padded_omega, inverse=False)
    
    # Pointwise multiply
    for j in range(padded_size):
        scalars_padded[j] = (scalars_padded[j] * g_inv_padded[j]) % modulus
    
    # Inverse FFT
    scalars_padded = fft_iterative(scalars_padded, modulus, padded_omega, inverse=True)
    
    # Extract result and rescale
    for j in range(n):
        scalars[j] = (scalars_padded[j] * g[j]) % modulus
    
    if inverse:
        # Apply inverse scaling
        n_inv = pow(n, modulus - 2, modulus)
        for j in range(n):
            scalars[j] = (scalars[j] * n_inv) % modulus
    
    return scalars


def _compute_rader_tables_127(modulus, omega_127):
    """
    Precompute the Rader permutation tables and FFT'd kernel for 127-point DFT.
    
    Rader's algorithm converts a prime-length-p DFT (here p=127) into a cyclic
    convolution of length p-1=126. The cyclic convolution is computed via linear
    convolution using FFT: we zero-pad to 256 (next power-of-2 >= 2*126-1=251).
    
    A primitive root of Z/127Z is g=3 (3 generates the multiplicative group mod 127).
    
    Args:
        modulus: The field prime
        omega_127: Primitive 127th root of unity in the field
        
    Returns:
        Tuple (perm_input, perm_output, kernel_fft, omega_pad, conv_len, pad_size)
        - perm_input[k] = g^k mod 127 for k=0..125  (input permutation indices)
        - perm_output[k] = g^(-k) mod 127 for k=0..125 (output permutation indices)
        - kernel_fft: FFT of the Rader kernel (size pad_size), precomputed
        - omega_pad: primitive pad_size-th root of unity
        - conv_len: cyclic convolution length (126)
        - pad_size: FFT padding size (256)
    """
    p = 127
    g = 3  # primitive root of Z/127Z
    conv_len = p - 1  # 126
    
    # Pad size: next power of 2 >= 2*conv_len - 1 = 251 → 256
    pad_size = 256
    
    # Build g^k mod 127 table (input permutation) and g^(-k) mod 127 (output permutation)
    perm_input = [0] * conv_len   # length 126
    perm_output = [0] * conv_len  # length 126
    
    g_power = 1
    g_inv = pow(g, p - 2, p)  # g^(-1) mod 127
    g_inv_power = 1
    
    for k in range(conv_len):
        perm_input[k] = g_power
        perm_output[k] = g_inv_power
        g_power = (g_power * g) % p
        g_inv_power = (g_inv_power * g_inv) % p
    
    # Build the Rader kernel: b[k] = omega_127^(g^(-k) mod 127) for k=0..125
    kernel = [0] * pad_size  # zero-pad 126 → 256
    for k in range(conv_len):
        kernel[k] = pow(omega_127, perm_output[k], modulus)
    
    # Compute omega_pad (primitive 256th root of unity in the field)
    # Since p-1 = 2^24 * 127, and 256 = 2^8 divides 2^24, this exists
    omega_pad = pow(3, (modulus - 1) // pad_size, modulus)
    
    # Precompute FFT of the kernel (can be reused for all 127-DFTs with same omega_127)
    kernel_fft = list(kernel)
    cooley_tukey_fft(kernel_fft, modulus, omega_pad, inverse=False)
    
    # Also precompute the inverse kernel FFT (for inverse DFTs)
    omega_inv = pow(omega_127, modulus - 2, modulus)
    inv_kernel = [0] * pad_size
    for k in range(conv_len):
        inv_kernel[k] = pow(omega_inv, perm_output[k], modulus)
    inv_kernel_fft = list(inv_kernel)
    cooley_tukey_fft(inv_kernel_fft, modulus, omega_pad, inverse=False)
    
    return perm_input, perm_output, kernel_fft, inv_kernel_fft, omega_pad, conv_len, pad_size


def rader_127_dft(values, modulus, omega_127, rader_tables, inverse=False):
    """
    Compute a 127-point DFT using Rader's algorithm.
    
    Rader's algorithm converts a prime-length-p DFT into a cyclic convolution
    of length p-1 = 126. We zero-pad to 256 and use Cooley-Tukey FFT for the
    linear convolution, then fold the result back to 126 for the cyclic result.
    
    Cost: ~3 × 256 × 8 + O(256) ≈ 6,400 multiplications
    vs. naive DFT matrix: 127² = 16,129 multiplications (~2.5× speedup)
    
    Args:
        values: List of 127 field elements
        modulus: Prime field modulus
        omega_127: Primitive 127th root of unity
        rader_tables: Precomputed (perm_input, perm_output, kernel_fft, inv_kernel_fft, omega_pad, conv_len, pad_size)
        inverse: If True, compute inverse DFT
        
    Returns:
        List of 127 transformed values
    """
    p = 127
    assert len(values) == p
    _mod = modulus  # local ref
    
    perm_input, perm_output, kernel_fft, inv_kernel_fft, omega_pad, conv_len, pad_size = rader_tables
    
    # Select forward or inverse kernel (both precomputed)
    active_kernel_fft = inv_kernel_fft if inverse else kernel_fft
    
    # Step 1: X[0] = sum of all input values
    x0 = sum(values) % _mod
    
    # Step 2: Permute input according to Rader's permutation
    # a[k] = values[g^k mod 127] for k = 0..125
    a = [0] * pad_size  # zero-pad 126 → 256
    for k in range(conv_len):
        a[k] = values[perm_input[k]]
    
    # Step 3: Linear convolution via FFT (we'll fold to cyclic afterwards)
    # linear_conv = IFFT( FFT(a) ⊙ FFT(kernel) )
    a_fft = list(a)
    cooley_tukey_fft(a_fft, _mod, omega_pad, inverse=False)
    
    # Pointwise multiply
    conv_fft = [a_fft[i] * active_kernel_fft[i] % _mod for i in range(pad_size)]
    
    # Inverse FFT
    cooley_tukey_fft(conv_fft, _mod, omega_pad, inverse=True)
    
    # Step 4: Fold linear convolution to cyclic convolution mod conv_len
    # cyclic_conv[k] = sum of linear_conv[k + j*conv_len] for all valid j
    cyclic_conv = [0] * conv_len
    for i in range(pad_size):
        if conv_fft[i] != 0:
            cyclic_conv[i % conv_len] = (cyclic_conv[i % conv_len] + conv_fft[i]) % _mod
    
    # Step 5: Build output
    # X[0] = sum(values)
    # X[g^(-s) mod 127] = values[0] + cyclic_conv[s] for s = 0..125
    result = [0] * p
    result[0] = x0
    v0 = values[0]
    
    for s in range(conv_len):
        result[perm_output[s]] = (v0 + cyclic_conv[s]) % _mod
    
    if inverse:
        # Scale by 1/127
        p_inv = pow(p, _mod - 2, _mod)
        for i in range(p):
            result[i] = result[i] * p_inv % _mod
    
    return result


def mixed_radix_127_fft(values, modulus, omega, inverse=False):
    """
    Mixed-Radix Cooley-Tukey FFT for sizes N = 127 * 2^k.
    
    Decomposes into:
    1) 2^k independent 127-point DFTs (using Rader's algorithm, ~5.5× faster than naive)
    2) Twiddle factor multiplication
    3) 127 independent 2^k-point FFTs (using Cooley-Tukey)
    
    Total cost: O(2^k × 2944 + 127 × 2^k × k) vs old O(2^k × 16129 + 127 × 2^k × k)
    """
    n = len(values)
    if n % 127 != 0:
        raise ValueError(f"Size must be a multiple of 127, got {n}")
        
    n1 = 127
    n2 = n // 127
    
    if (n2 & (n2 - 1)) != 0 and n2 != 1:
        raise ValueError(f"Size/127 must be a power of 2, got {n2}")
        
    if inverse:
        omega = pow(omega, modulus - 2, modulus)
        
    omega_n1 = pow(omega, n2, modulus)   # primitive 127th root
    omega_n2 = pow(omega, n1, modulus)   # primitive 2^k-th root
    
    # Precompute Rader tables for 127-point DFTs (shared across all n2 columns)
    rader_tables = _compute_rader_tables_127(modulus, omega_n1)
    
    # Precompute twiddle factors for the combination step
    twiddles_n = [1] * n
    for i in range(1, n):
        twiddles_n[i] = (twiddles_n[i-1] * omega) % modulus
        
    # Step 1: n2 independent 127-point DFTs using Rader's algorithm
    # Input is stored as: values[n1_idx * n2 + n2_idx]
    # For each n2_idx, extract column and apply 127-point DFT
    
    # Initialize Z as a list of n1 lists of size n2
    Z = [[0] * n2 for _ in range(n1)]
    
    for n_2 in range(n2):
        # Extract the column: x(n1*N2 + n2)
        col_127 = [values[n_1 * n2 + n_2] for n_1 in range(n1)]
        
        # Apply 127-point DFT using Rader's algorithm
        col_dft = rader_127_dft(col_127, modulus, omega_n1, rader_tables, inverse=False)
        
        # Apply cross-twiddle factors and store
        for k1 in range(n1):
            twiddle_idx = (k1 * n_2) % n
            Z[k1][n_2] = (col_dft[k1] * twiddles_n[twiddle_idx]) % modulus
            
    # Step 2: n1 independent n2-point FFTs using Cooley-Tukey
    result = [0] * n
    
    for k1 in range(n1):
        row_fft = list(Z[k1])
        if n2 > 1:
            cooley_tukey_fft(row_fft, modulus, omega_n2, inverse=False)
            
        # Place into output array X(k2*N1 + k1)
        for k2 in range(n2):
            result[k2 * n1 + k1] = row_fft[k2]
            
    if inverse:
        # Scale the result by 1/N
        n_inv = pow(n, modulus - 2, modulus)
        for i in range(n):
            result[i] = (result[i] * n_inv) % modulus
            
    return result


# Backward compatibility wrapper
def fft(vals, modulus, root_of_unity, inv=False):
    """
    FFT wrapper using Cooley-Tukey for power-of-2 sizes,
    Bluestain for other sizes.
    
    Args:
        vals: Values to transform
        modulus: Prime field modulus
        root_of_unity: n-th root of unity where n = len(vals)
        inv: If True, compute inverse FFT
    
    Returns:
        Transformed values
    """
    values = list(vals)
    n = len(values)
    
    # Size 1: no-op
    if n == 1:
        return values
    
    # Size 0: empty
    if n == 0:
        return values
    
    # Check if power of 2
    if (n & (n - 1)) == 0:
        # Power-of-2: use Cooley-Tukey
        values_out = list(values)
        cooley_tukey_fft(values_out, modulus, root_of_unity, inverse=inv)
        return values_out
    elif n % 127 == 0 and ((n // 127) & ((n // 127) - 1)) == 0:
        # Size is 127 * 2^k: use true mixed radix
        return mixed_radix_127_fft(values, modulus, root_of_unity, inverse=inv)
    else:
        # Other sizes: use Bluestain
        return bluestein_ntt(values, modulus, root_of_unity, inverse=inv)
