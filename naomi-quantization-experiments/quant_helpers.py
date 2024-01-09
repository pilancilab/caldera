import numpy as np
from scipy import stats
from sklearn.utils.extmath import randomized_svd as rand_svd
import matplotlib.pyplot as plt
# svd = np.linalg.svd
svd = rand_svd

def naive_quant(A, A_max=None, b=4, epsilon=1e-8):
    if A_max is None:
        A_max = np.max(np.abs(A))
    else: 
        A = np.maximum(np.minimum(A, A_max), -A_max)
    
    A_max = np.maximum(A_max, epsilon)

    delta = A_max / (2**b - 1)
    return A // delta * delta

def nf_values(A_max, b=4):
    delta = 1/2 * (1/30 + 1/32)
    c1 = (1/2 - delta) / (2**(b-1) - 1)
    c2 = (1/2 - delta) / (2**(b-1))

    q_neg = stats.norm.ppf(c1 * np.arange(2**(b-1)) + delta) / stats.norm.ppf(1-delta)
    q_pos = stats.norm.ppf(c2 * np.arange(2**(b-1) + 1) + 1/2) / stats.norm.ppf(1-delta)
    return A_max * np.hstack((q_neg[:-1], q_pos))

def nf_quant(A, b=4, A_max=None, epsilon=1e-8):
    assert b >= 2, "NF quantization only works for >= 2 bits"
    if A_max is not None:
        A = np.maximum(np.minimum(A, A_max), -A_max)
    else:
        A_max = np.max(np.abs(A))

    A_max = np.maximum(A_max, epsilon)
    two_to_b_minus_1 = int(np.floor(2**(b-1)))
    delta = 1/2 * (1/30 + 1/32)
    c1 = (1/2 - delta) / (two_to_b_minus_1 - 1)
    c2 = (1/2 - delta) / two_to_b_minus_1

    A_norm = A / A_max
    q_neg = stats.norm.ppf(c1 * np.arange(two_to_b_minus_1) + delta) / stats.norm.ppf(1-delta)
    q_pos = stats.norm.ppf(c2 * np.arange(two_to_b_minus_1 + 1) + 1/2) / stats.norm.ppf(1-delta)
    
    neg_quant_idxs = (A < 0) * np.round((stats.norm.cdf(A_norm * stats.norm.ppf(1-delta)) - delta) / c1).astype(int)
    pos_quant_idxs = (A >= 0) * np.round((stats.norm.cdf(A_norm * stats.norm.ppf(1-delta)) - 1/2) / c2).astype(int)

    A_quant_norm = (A < 0) * q_neg[neg_quant_idxs] + (A >= 0) * q_pos[pos_quant_idxs]
    return A_quant_norm * A_max

def dithered_nf_quant(A, b=4, A_max=None, epsilon=1e-8):
    assert b >= 2, "NF quantization only works for >= 2 bits"
    if A_max is not None:
        A = np.maximum(np.minimum(A, A_max), -A_max)
    else:
        A_max = np.max(np.abs(A))

    A_max = np.maximum(A_max, epsilon)
    two_to_b_minus_1 = int(np.floor(2**(b-1)))
    delta = 1/2 * (1/30 + 1/32)
    c1 = (1/2 - delta) / (two_to_b_minus_1 - 1)
    c2 = (1/2 - delta) / two_to_b_minus_1

    A_norm = A / A_max
    q_neg = stats.norm.ppf(c1 * np.arange(two_to_b_minus_1) + delta) / stats.norm.ppf(1-delta)
    q_pos = stats.norm.ppf(c2 * np.arange(two_to_b_minus_1 + 1) + 1/2) / stats.norm.ppf(1-delta)
    
    neg_cdf_vals = (stats.norm.cdf(A_norm * stats.norm.ppf(1-delta)) - delta) / c1
    neg_cdf_round_down = np.floor(neg_cdf_vals).astype(int)
    neg_cdf_round_up = np.ceil(neg_cdf_vals).astype(int)
    neg_p = np.nan_to_num((neg_cdf_vals - neg_cdf_round_down) / (neg_cdf_round_up - neg_cdf_round_down))
    neg_quant_idxs = (A < 0) * bernoulli(neg_p, neg_cdf_round_down, neg_cdf_round_up)

    pos_cdf_vals = (stats.norm.cdf(A_norm * stats.norm.ppf(1-delta)) - 1/2) / c2
    pos_cdf_round_down = np.floor(pos_cdf_vals).astype(int)
    pos_cdf_round_up = np.ceil(pos_cdf_vals).astype(int)
    pos_p = np.nan_to_num((pos_cdf_vals - pos_cdf_round_down) / (pos_cdf_round_up - pos_cdf_round_down))
    pos_quant_idxs = (A >= 0) * bernoulli(pos_p, pos_cdf_round_down, pos_cdf_round_up)

    A_quant_norm = (A < 0) * q_neg[neg_quant_idxs] + (A >= 0) * q_pos[pos_quant_idxs]
    return A_quant_norm * A_max

def bernoulli(p, opt_0=0, opt_1=1):
    size = (1, )
    if hasattr(opt_0, "__array__"):
        size = opt_0.shape

    opt_choice = np.random.rand(*size) < p
    return opt_choice * opt_1 + np.bitwise_not(opt_choice) * opt_0

def quant_blocked(A, b=4, block_size=64, quant_fn=nf_quant):
    A_reshaped = A.flatten()
    zeros_added = (A_reshaped.size + block_size - 1) // block_size * block_size - A_reshaped.size
    A_reshaped = np.hstack((A_reshaped, np.zeros(zeros_added)))
    A_reshaped = A_reshaped.reshape((block_size, A_reshaped.size // block_size))

    A_max = np.max(A_reshaped, axis=0)

    A_reshaped_quant = quant_fn(A_reshaped, b=b, A_max=A_max)
    return A_reshaped_quant.flatten()[:A_reshaped.size-zeros_added].reshape(A.shape)

def nf_quant_blocked(A, **kwargs):
    return quant_blocked(A, quant_fn=nf_quant, **kwargs)

def nf_quant_factors(A, B, prod=None, num_tries=10, b=4, block_size=64):
    if prod is not None:
        W = prod
    else:
        W = A @ B

    QA = nf_quant_blocked(A, b=b, block_size=block_size)
    QB = nf_quant_blocked(B, b=b, block_size=block_size)
    best_quant = (A, B)
    best_err = np.linalg.norm(W - QA @ QB, 'fro')
    for i in range(num_tries):
        QA = quant_blocked(A, b=b, block_size=block_size, quant_fn=dithered_nf_quant)
        QB = quant_blocked(B, b=b, block_size=block_size, quant_fn=dithered_nf_quant)

        err = np.linalg.norm(W - QA @ QB, 'fro')
        if err < best_err:
            best_quant = (QA, QB)
            best_err = err
    return best_quant

def shrinkage(tau, x):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def inexact_RPCA(W, k_max, lmbda, print_freq=5, n_iter=10, rtol=1e-6, mu=None):
    ### Like RPCA, but having W = L + S is part of the objective, not the constraint.
    ### This lets us get a slightly inaccurate representation in exchange for better
    ### sparsity or lower rank
    if mu is None:
        mu = np.linalg.norm(W, 'fro') / (4 * np.sum(np.abs(W)))
    L = W
    S = np.zeros_like(W)

    lmbda *= np.linalg.norm(W, 'fro') / (W.shape[0]*W.shape[1])
    
    print(f"\t[DEBUG] Initial Error: {np.linalg.norm(W - L - S, 'fro')}")
    print(f"\t[DEBUG] Initial Rank: {min(W.shape[0], W.shape[1])}")

    # Y1 = np.zeros_like(W)

    for j in range(n_iter):
        if j % print_freq == 0:
            print(f"\t[DEBUG] {j}")

        ## Set L
        # U, Sigma, VT = svd(W - S - 1/mu * Y1, k_max)
        U, Sigma, VT = svd(W - S, k_max)
        Sigma = np.diag(shrinkage(mu, Sigma))
        L = U @ Sigma @ VT

        rank = np.sum(Sigma > 0)

        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set L): {np.linalg.norm(W - L - S, 'fro')}")
            print(f"\t\t[DEBUG] Rank: {rank}\n")

        ## Set S
        S = shrinkage(lmbda/mu, W - L)

        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set S): {np.linalg.norm(W - L - S, 'fro')}")

        ## Set Y
        # Y1 += mu * (W - L - S)

    return L, S

def make_sparse(A, n_to_keep):
    A_flat = np.abs(A.flatten())
    nth_largest = A_flat[np.argpartition(-A_flat, n_to_keep)[n_to_keep]]
    return A * (np.abs(A) > nth_largest)