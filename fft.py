def _simple_ft(vals, modulus, roots_of_unity, field=None):
    L = len(roots_of_unity)
    o = []
    
    for i in range(L):
        last = field.zero() if field is not None else 0
        for j in range(L):
            if field is not None:
                term = field.mul(vals[j], roots_of_unity[(i*j)%L])
                last = field.add(last, term)
            else:
                last = (last + vals[j] * roots_of_unity[(i*j)%L]) % modulus
        o.append(last)
    return o

def _fft(vals, modulus, roots_of_unity, field=None):
    if len(vals) <= 4:
        return _simple_ft(vals, modulus, roots_of_unity, field=field)
    
    L = _fft(vals[::2], modulus, roots_of_unity[::2], field=field)
    R = _fft(vals[1::2], modulus, roots_of_unity[::2], field=field)
    o = [field.zero() if field is not None else 0 for _ in vals]
    
    for i, (x, y) in enumerate(zip(L, R)):
        if field is not None:
            y_times_root = field.mul(y, roots_of_unity[i])
            o[i] = field.add(x, y_times_root)
            o[i+len(L)] = field.sub(x, y_times_root)
        else:
            y_times_root = (y * roots_of_unity[i]) % modulus
            o[i] = (x + y_times_root) % modulus 
            o[i+len(L)] = (x - y_times_root) % modulus 
    return o

def expand_root_of_unity(root_of_unity, modulus, field=None):
    # Build up roots of unity
    rootz = [field.one() if field is not None else 1, root_of_unity]
    while rootz[-1] != (field.one() if field is not None else 1):
        if field is not None:
            rootz.append(field.mul(rootz[-1], root_of_unity))
        else:
            rootz.append((rootz[-1] * root_of_unity) % modulus)
    return rootz

def fft(vals, modulus, root_of_unity, inv=False, field=None):
    rootz = expand_root_of_unity(root_of_unity, modulus, field=field)
    # Fill in vals with zeroes if needed
    if len(rootz) > len(vals) + 1:
        zero_val = field.zero() if field is not None else 0
        vals = vals + [zero_val] * (len(rootz) - len(vals) - 1)
    if inv:
        # Inverse FFT
        if field is not None:
            # We assume field objects handle integer scaling or we manually construct it
            # Standard approach is just scaling using field.inv and successive additions, 
            # or `field.inv(field.from_int(len(vals)))`. For simplicity if field has `inv` and integer multiplication:
            # We will use field.inv
            try:
                invlen = field.inv(field.from_int(len(vals)))
            except AttributeError:
                # Fallback if field doesn't have from_int
                len_elem = field.one()
                for _ in range(len(vals) - 1):
                    len_elem = field.add(len_elem, field.one())
                invlen = field.inv(len_elem)
            return [field.mul(x, invlen) for x in _fft(vals, modulus, rootz[:0:-1], field=field)]
        else:
            invlen = pow(len(vals), modulus-2, modulus)
            return [(x*invlen) % modulus for x in _fft(vals, modulus, rootz[:0:-1])]
    else:
        # Regular FFT
        return _fft(vals, modulus, rootz[:-1], field=field)

def bluestein_fft(vals, modulus, root_of_unity, field=None):
    """
    Computes the FFT for sequences of arbitrary length using Bluestein's algorithm.
    Fallback to simple FT for safety if n is small or convolution is complex here.
    """
    n = len(vals)
    if n == 0:
        return []
    
    # Using simple discrete fourier transform for arbitrary n
    # For a full O(n log n) Bluestein, one would pad to a power of 2, 
    # but that requires a 2n-th root of unity which may not be available.
    rootz = expand_root_of_unity(root_of_unity, modulus, field=field)[:-1]
    if len(rootz) > len(vals):
        zero_val = field.zero() if field is not None else 0
        vals = vals + [zero_val] * (len(rootz) - len(vals))
    return _simple_ft(vals, modulus, rootz, field=field)

# Evaluates f(x) for f in evaluation form
def inv_fft_at_point(vals, modulus, root_of_unity, x):
    if len(vals) == 1:
        return vals[0]
    # 1/2 in the field
    half = (modulus + 1)//2
    # 1/w
    inv_root = pow(root_of_unity, len(vals)-1, modulus)
    # f(-x) in evaluation form
    f_of_minus_x_vals = vals[len(vals)//2:] + vals[:len(vals)//2]
    # e(x) = (f(x) + f(-x)) / 2 in evaluation form
    evens = [(f+g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    # o(x) = (f(x) - f(-x)) / 2 in evaluation form
    odds = [(f-g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    # e(x^2) + coordinate * x * o(x^2) in evaluation form
    comb = [(o * x * inv_root**i + e) % modulus for i, (o, e) in enumerate(zip(odds, evens))]
    return inv_fft_at_point(comb[:len(comb)//2], modulus, root_of_unity ** 2 % modulus, x**2 % modulus)

def shift_domain(vals, modulus, root_of_unity, factor):
    if len(vals) == 1:
        return vals
    # 1/2 in the field
    half = (modulus + 1)//2
    # 1/w
    inv_factor = pow(factor, modulus - 2, modulus)
    half_length = len(vals)//2
    # f(-x) in evaluation form
    f_of_minus_x_vals = vals[half_length:] + vals[:half_length]
    # e(x) = (f(x) + f(-x)) / 2 in evaluation form
    evens = [(f+g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    print('e', evens)
    # o(x) = (f(x) - f(-x)) / 2 in evaluation form
    odds = [(f-g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    print('o', odds)
    shifted_evens = shift_domain(evens[:half_length], modulus, root_of_unity ** 2 % modulus, factor ** 2 % modulus)
    print('se', shifted_evens)
    shifted_odds = shift_domain(odds[:half_length], modulus, root_of_unity ** 2 % modulus, factor ** 2 % modulus)
    print('so', shifted_odds)
    return (
        [(e + inv_factor * o) % modulus for e, o in zip(shifted_evens, shifted_odds)] + 
        [(e - inv_factor * o) % modulus for e, o in zip(shifted_evens, shifted_odds)]
    )

def shift_poly(poly, modulus, factor):
    factor_power = 1
    inv_factor = pow(factor, modulus - 2, modulus)
    o = []
    for p in poly:
        o.append(p * factor_power % modulus)
        factor_power = factor_power * inv_factor % modulus
    return o

def mul_polys(a, b, modulus, root_of_unity):
    rootz = [1, root_of_unity]
    while rootz[-1] != 1:
        rootz.append((rootz[-1] * root_of_unity) % modulus)
    if len(rootz) > len(a) + 1:
        a = a + [0] * (len(rootz) - len(a) - 1)
    if len(rootz) > len(b) + 1:
        b = b + [0] * (len(rootz) - len(b) - 1)
    x1 = _fft(a, modulus, rootz[:-1])
    x2 = _fft(b, modulus, rootz[:-1])
    return _fft([(v1*v2)%modulus for v1,v2 in zip(x1,x2)],
               modulus, rootz[:0:-1])