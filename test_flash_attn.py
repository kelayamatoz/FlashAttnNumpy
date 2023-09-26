# %%
import numpy as np
from scipy.special import softmax
# %%
N = 256
d = 8
M = 64

Q = np.random.random((N, d))
K = np.random.random((N, d))
V = np.random.random((N, d))

# %%
def standard_attn(Q, K, V):
	"""
	Standard attentin as described in Algo. 0.
	"""
	S = Q.dot(K.T)
	P = softmax(S, axis=1)
	O = P.dot(V)
	return O

# %%
O = standard_attn(Q, K, V)
print("First [:4, :4] matrix of standard attention = {}".format(O[:4, :4]))
# %%
def flash_attn(
	Q, K, V, O, Lse, BLOCK_M, BLOCK_DMODEL, BLOCK_N
):
	"""
	I'm following triton implementation from here: 
	https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
	"""
	lo, hi = 0, N
	for start_n in range(lo, hi, BLOCK_N):
		# --- compute qk ---
		k = K[start_n : start_n + BLOCK_N]
		qk = np.zeros((BLOCK_M, BLOCK_N))

