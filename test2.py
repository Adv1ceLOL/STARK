from poly_utils import ExtensionField
f_ext = ExtensionField(2**31-1)
G2 = f_ext.exp((3, 1), ((2**31-1)**2 - 1) // (8192 * 4))
print('G2 =', G2)
G1 = f_ext.exp(G2, 4)
print('G1 =', G1)
rootz = [f_ext.one(), G1]
print(f_ext.one() == (1,0))
c = 0
while rootz[-1] != f_ext.one():
    rootz.append(f_ext.mul(rootz[-1], G1))
    c += 1
    if c > 20000:
        print('Infinite loop!')
        break
print('Done, len:', len(rootz))
