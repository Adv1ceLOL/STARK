import argparse

P = 2**31 - 2**24 + 1
DEFAULT_RF = 4
DEFAULT_RP = 1
DEFAULT_ALPHA = 3
DEFAULT_T = 16
ROUND_CONSTANTS = [
	1337955004,
	1367429177,
	1743914851,
	740223888,
	1468216011,
	1719134033,
	453365459,
	530345905,
	721433499,
	1509841703,
	1825779369,
	659445153,
	1761978132,
	951227651,
	805393251,
	2121768148,
	1580739153,
	699788269,
	575751146,
	1487572327,
	636228153,
	1235912527,
	1799708221,
	627223601,
	1985896975,
	232455476,
	1552660474,
	1286504011,
	133305388,
	1330250169,
	1873351758,
	1249254749,
	656656135,
	1215211643,
	2058118293,
	265689576,
	1593649773,
	560890292,
	1892799583,
	655375650,
	2033641140,
	77722953,
	639997731,
	93592034,
	136143744,
	17163270,
	1902294035,
	1481572108,
	1365459094,
	688048860,
	1470115723,
	1500521211,
	1680922619,
	1272262914,
	1275388246,
	844784829,
	1706658472,
	2051757360,
	1366089976,
	77633995,
	632856853,
	364154585,
	256285265,
	181966044,
	1182591191,
]
MATRIX_FULL = [
	[10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3],
	[8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1],
	[2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7],
	[2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6],
	[5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3],
	[4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1],
	[1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7],
	[1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6],
	[5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3],
	[4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1],
	[1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7],
	[1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6],
	[5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6],
	[4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2],
	[1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14],
	[1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12],
]
MATRIX_PARTIAL = [
	[1329827747, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 875786758, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 132889202, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1074162980, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1766999771, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 492235992, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 2061343503, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1517024595, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 534738949, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 838363649, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 254279956, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1553366804, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1756173309, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1473904101, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 628009981, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1492201877],
]

#Matrix Vector Multiplication
def mat_vec_mul(matrix: list[list[int]], vector: list[int], modulus: int) -> list[int]:
	return [sum((a * b) % modulus for a, b in zip(row, vector)) % modulus for row in matrix]

def poseidon2(
	state: list[int], rc: list[int], matrix_full: list[list[int]], matrix_partial: list[list[int]],
	modulus: int, rf: int, rp: int, alpha: int,
	) -> list[int]:
	t = len(state)

	state_words = [x % modulus for x in state] #finite field arithmetic
	state_words = mat_vec_mul(matrix_full, state_words, modulus) #initial mixing layer

	rc_iter = iter(rc)
	
	def full_round() -> None:
		for i in range(len(state_words)):
			state_words[i] = (state_words[i] + next(rc_iter)) % modulus
		for i in range(t): #iteration of non-linear component
			state_words[i] = pow(state_words[i], alpha, modulus) 
		state_words[:] = mat_vec_mul(matrix_full, state_words, modulus)

	def partial_round() -> None:
		state_words[0] = (state_words[0] + next(rc_iter)) % modulus
		state_words[0] = pow(state_words[0], alpha, modulus)
		state_words[:] = mat_vec_mul(matrix_partial, state_words, modulus)
	
	half_rf = rf // 2
	
	#Rounds
	for _ in range(half_rf):
		full_round()

	for _ in range(rp):
		partial_round()

	for _ in range(half_rf):
		full_round()

	return state_words

#Vector translator
def parse_state_arg(value: str, t: int) -> list[int]:
	parts = [p.strip() for p in value.split(",") if p.strip()]
	return [int(x) for x in parts]

def main() -> None:
    print("Running Poseidon2 with default parameters...")

    rc = ROUND_CONSTANTS
    me = MATRIX_FULL
    mi = MATRIX_PARTIAL
    t = DEFAULT_T

    state_input_str = (
        "1253653436,1786871594,1305924322,1544663848,1581949976,1400364434,754998583,"
        "2043969975,185533518,2025969807,994876701,1122188176,1954375908,724240441,0,0"
    )
    state = parse_state_arg(state_input_str, t=t)

    output = poseidon2(
        state=state,
        rc=rc,
        matrix_full=me,
        matrix_partial=mi,
        modulus=P,
        rf=DEFAULT_RF,
        rp=DEFAULT_RP,
        alpha=DEFAULT_ALPHA,
    )

    print("Input:")
    print(state)
    print("Output:")
    print(output)

if __name__ == "__main__":
    main()

