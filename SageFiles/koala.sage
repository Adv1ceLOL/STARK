import os
p = 2**31 -2**24+1
F = GF(p)
R_F_FIXED = 6
RF = R_F_FIXED
R_P_FIXED = 4 
RP = R_P_FIXED
d = 3
t = 16
load("poseidon2_rust_params.sage")
MI = MATRIX_PARTIAL
ME = MATRIX_FULL
rc = round_constants

print("Round Costants: ")
print(rc)
print("Matrix Full: ")
print(ME)
print("Matrix Partial: ")
print(MI)

def ARC(v, R):
    for i in range(t):
        v[i] += rc[t*R + i]
    return v

def S_full(v):
    for i in range(t):
        v[i] = v[i]**d
    return v

def S_partial(v):
    v[0] = v[0]**d
    return v

def Poseidon(v):
    ret = deepcopy(v)
    ret = ME * ret
    for i in range(RF//2):
        ret = ARC(ret, i)
        ret = S_full(ret)
        ret = ME * ret
    for i in range(RP):
        ret = ARC(ret, RF//2 + i)
        ret = S_partial(ret)
        ret = MI * ret
    for i in range(RF//2):
        ret = ARC(ret, RF//2 + RP + i)
        ret = S_full(ret)
        ret = ME * ret
    return ret


v = vector(F, (1253653436, 1786871594, 1305924322, 1544663848, 1581949976, 1400364434, 754998583, 2043969975, 185533518, 2025969807, 994876701, 1122188176, 1954375908, 724240441, 0, 0))

print("Input:")
print(v)
print("Output:")
print(Poseidon(v))



