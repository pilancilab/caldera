import numpy as np
from scipy import stats
import torch
from sklearn.utils.extmath import randomized_svd as rand_svd
import matplotlib.pyplot as plt

from quant_helpers import *
from quant_opt import *

# svd = np.linalg.svd
svd = rand_svd

def quantize_LQLORA(W, k, b=4, print_freq=5, n_iter=20, stop_tolerance=1e-3):
    Q = np.zeros_like(W)

    best_err = float('inf')
    best_mtxs = ()
    for i in range(n_iter):
        if i % print_freq == 0:
            print(f"\t[DEBUG] {i}")
        U, S, VT = svd(W - Q, k)
        U = U[:, :k]
        V = VT.T[:, :k]
        S = np.diag(S[:k])

        L = U @ np.sqrt(S)
        R = V @ np.sqrt(S)

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set L, R): {np.linalg.norm(W  - L @ R.T- Q, 'fro')}")

        Q_old = np.copy(Q)
        prev_err = np.linalg.norm(W  - L @ R.T- Q, 'fro')
        Q = nf_quant_blocked(W - L @ R.T, b=b)
        if prev_err < np.linalg.norm(W  - L @ R.T- Q, 'fro'):
            Q = Q_old

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set Q): {np.linalg.norm(W  - L @ R.T- Q, 'fro')}")

        err = np.linalg.norm(W - L @ R.T - Q, 'fro')
        prev_best_err = best_err
        if err < best_err:
            best_err = err
            best_mtxs = (L, R, Q)
        if prev_best_err - err < stop_tolerance:
            break

    return best_err, best_mtxs

def quantize_LRQ1Q2(W, k1, k2, b=4, keep_S=False, print_freq=5, epsilon=1e-2, n_iter_outer=20, stop_tolerance=1e-3):
    U, S, VT = svd(W, k1+k2)
    V = VT.T
    S = np.diag(S)

    L = U[:, :k1] @ np.sqrt(S[:k1, :k1])
    R = V[:, :k1] @ np.sqrt(S[:k1, :k1])

    SQ = S[k1:, k1:]
    if keep_S:
        Q1_unquant = U[:, k1:]
        Q2_unquant = V[:, k1:]
        SQ = S[k1:, k1:]
    else:
        Q1_unquant = U[:, k1:] @ np.sqrt(S[k1:, k1:])
        Q2_unquant = V[:, k1:] @ np.sqrt(S[k1:, k1:])

    get_error = lambda L, R, Q1, Q2, S: np.linalg.norm(W - L@R.T - Q1 @ Q2.T, 'fro') if not keep_S \
                                     else np.linalg.norm(W - L@R.T - Q1 @ S @ Q2.T, 'fro')

    Q1 = nf_quant_blocked(Q1_unquant, b=b)
    Q2 = nf_quant_blocked(Q2_unquant, b=b)

    print(f"\t[DEBUG] Frobenius norm of W - W_hat (initial): {get_error(L, R, Q1, Q2, SQ)}")

    best_err = float('inf')
    best_mtxs = ()

    for i in range(n_iter_outer):
        if i % print_freq == 0:
            print(f"\t[DEBUG] {i}")

        for _ in range(10):
            err1 = get_error(L, R, Q1, Q2, SQ)
            Q1_old, Q2_old = np.copy(Q1), np.copy(Q2)

            if keep_S:
                Q1 = nf_quant_blocked(np.linalg.lstsq(Q2 @ SQ, W.T - R @ L.T, rcond=None)[0].T, b=b)
                Q2 = nf_quant_blocked(np.linalg.lstsq(Q1 @ SQ, W - L @ R.T, rcond=None)[0].T, b=b)
            else:
                Q1 = nf_quant_blocked(np.linalg.lstsq(Q2, W.T - R @ L.T, rcond=None)[0].T, b=b)
                Q2 = nf_quant_blocked(np.linalg.lstsq(Q1, W - L @ R.T, rcond=None)[0].T, b=b)

            err2 = get_error(L, R, Q1, Q2, SQ)
            if err1 < err2:
                Q1, Q2 = Q1_old, Q2_old # restore the old values if this iteration made things worse

            if err1 - err2 < epsilon:
                break

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set Q1, Q2): {get_error(L, R, Q1, Q2, SQ)}")

        U, S, VT = svd(W - Q1 @ Q2.T, k1)
        S = np.diag(S)
        V = VT.T

        L = U @ np.sqrt(S)
        R = V @ np.sqrt(S)

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set L, R): {get_error(L, R, Q1, Q2, SQ)}")
        
        err = get_error(L, R, Q1, Q2, SQ)
        if err < best_err:
            best_err = err
            best_mtxs = (L, R, Q1, Q2, SQ)

    return best_err, best_mtxs


def quantize_LRQ1Q2_plus_sparse(W, k1, k2, lmbda=1e-1, desired_sparsity_ratio=0.25, b=4, print_freq=5, epsilon=1e-8, n_iter_outer=20, stop_tolerance=1e-3):
    print("Running RPCA...")
    LR, S = inexact_RPCA(W, k1+k2, lmbda)
    print("RPCA done.")
    U, Sigma, VT = svd(LR, k1+k2)
    Sigma = np.diag(Sigma)
    V = VT.T

    L = U[:, :k1] @ np.sqrt(Sigma[:k1, :k1])
    R = V[:, :k1] @ np.sqrt(Sigma[:k1, :k1])

    Q1_unquant = U[:, k1:] @ np.sqrt(Sigma[k1:, k1:])
    Q2_unquant = V[:, k1:] @ np.sqrt(Sigma[k1:, k1:])
    Q1 = nf_quant_blocked(Q1_unquant, b=b)
    Q2 = nf_quant_blocked(Q2_unquant, b=b)

    S = nf_quant(S, b=b)

    sparsity = np.sum(np.abs(S) > epsilon)
    print("Sparsity, relative sparsity of S:", sparsity, " ", sparsity / (S.shape[0]*S.shape[1]))
    sparsity = int(np.round(S.shape[0]*S.shape[1]*desired_sparsity_ratio))
    S = make_sparse(S, sparsity)
    print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (init): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

    for i in range(n_iter_outer):
        err_before = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')
        if i % print_freq == 0:
            print(f"\t[DEBUG] {i}")
        ## set S
        S = nf_quant_blocked(
            make_sparse(W - L @ R.T - Q1 @ Q2.T, sparsity),
            b=b, block_size=int(np.round(64/desired_sparsity_ratio)))
        # sparsity = np.sum(np.abs(S) > epsilon)
        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set S): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

        ## set Q1, Q2
        for _ in range(10):
            err1 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')
            Q1_old, Q2_old = np.copy(Q1), np.copy(Q2)
            Q1_old_unq, Q2_old_unq = np.copy(Q1_unquant), np.copy(Q2_unquant)

            Q1_unquant = np.linalg.lstsq(Q2, W.T - R @ L.T - S.T, rcond=None)[0].T
            Q1, Q2 = nf_quant_factors(Q1_unquant, Q2_unquant.T, b=b, prod=W - L @ R.T - S)
            Q2 = Q2.T
            Q1 = nf_quant_blocked(Q1_unquant, b=b)

            Q2_unquant = np.linalg.lstsq(Q1, W - L @ R.T - S, rcond=None)[0].T
            Q1, Q2 = nf_quant_factors(Q1_unquant, Q2_unquant.T, b=b, prod=W - L @ R.T - S)
            Q2 = Q2.T

            err2 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')

            if err1 < err2:
                Q1, Q2 = Q1_old, Q2_old # restore the old values if this iteration made things worse
                Q1_unquant, Q2_unquant = Q1_old_unq, Q2_old_unq

            if err1 - err2 < stop_tolerance:
                break

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set Q1, Q2): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

        ## set L, R
        U, Sigma, VT = svd(W - S - Q1 @ Q2.T, k1)
        Sigma = np.diag(Sigma)
        V = VT.T

        L = U[:, :k1] @ np.sqrt(Sigma[:k1, :k1])
        R = V[:, :k1] @ np.sqrt(Sigma[:k1, :k1])

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set L, R): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

        if err_before - np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro') < stop_tolerance:
            break

    m, n = W.shape
    num_bits = 1 * m*n + b * (m + n) * k2 + 32 * (m + n) * k1 + b * sparsity
    return num_bits, np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro'), (L, R, Q1, Q2, S)

def quantize_LQLORA_split(W, k1, k2, factor_b=5, b=4, print_freq=5, n_iter_outer=20, stop_tolerance=1e-3):
    U, S, VT = svd(W, k1+k2)
    V = VT.T
    S = np.diag(S)

    L = U[:, :k1] @ np.sqrt(S[:k1, :k1])
    R = V[:, :k1] @ np.sqrt(S[:k1, :k1])

    Q1_unquant = U[:, k1:] @ np.sqrt(S[k1:, k1:])
    Q2_unquant = V[:, k1:] @ np.sqrt(S[k1:, k1:])

    Q1 = nf_quant_blocked(Q1_unquant, b=factor_b)
    Q2 = nf_quant_blocked(Q2_unquant, b=factor_b)

    Q = nf_quant_blocked(W - L@R.T - Q1@Q2.T, b=b)

    print(f"\t[DEBUG] Frobenius norm of W - W_hat (initial): {np.linalg.norm(W - L@R.T - Q1 @ Q2.T - Q, 'fro')}")

    best_err = np.linalg.norm(W - L@R.T - Q1 @ Q2.T - Q, 'fro')
    best_mtxs = (L, R, Q1, Q2, Q)

    for i in range(n_iter_outer):
        if i % print_freq == 0:
            print(f"\t[DEBUG] {i}")

        for _ in range(10):
            err1 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - Q, 'fro')
            Q1_old, Q2_old = np.copy(Q1), np.copy(Q2)

            Q1 = nf_quant_blocked(np.linalg.lstsq(Q2, W.T - R @ L.T - Q, rcond=None)[0].T, b=factor_b)
            Q2 = nf_quant_blocked(np.linalg.lstsq(Q1, W - L @ R.T - Q, rcond=None)[0].T, b=factor_b)

            err2 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - Q, 'fro')

            if err1 < err2:
                Q1, Q2 = Q1_old, Q2_old # restore the old values if this iteration made things worse

            if err1 - err2 < stop_tolerance:
                break

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set Q1, Q2): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - Q, 'fro')}")

        Q = nf_quant_blocked(W - L@R.T - Q1@Q2.T, b=b)
        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set Q): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - Q, 'fro')}")

        U, S, VT = svd(W - Q1 @ Q2.T - Q, k1)
        S = np.diag(S)
        V = VT.T

        L = U @ np.sqrt(S)
        R = V @ np.sqrt(S)

        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set L, R): {np.linalg.norm(W  - L @ R.T- Q1 @ Q2.T - Q, 'fro')}")
        
        err = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - Q, 'fro')
        err_before = best_err
        if err < best_err:
            best_err = err
            best_mtxs = (L, R, Q1, Q2)

        if err_before - err < stop_tolerance:
            break

    return best_err, best_mtxs

def test_quant(W, b=3):
    print(f"Frobenius norm of W: {np.linalg.norm(W, 'fro')}")
    print("\tNumber of bits: %e\n" % (32*W.shape[0]*W.shape[1]))

    print(f"Error of Full NF Quantization ({b}B): {np.linalg.norm(W - nf_quant_blocked(W, b=b), 'fro')}")
    print("\tNumber of bits: %e\n" % (b*W.shape[0]*W.shape[1]))

    # quantize_noncvx(W, 260, 1e-3, b=b, n_iter=100, print_freq=5)

    # print("Testing LR + LR-Quant + Sparse")
    # k1 = 10
    # k2=500
    # num_bits, err, _ = quantize_LRQ1Q2_plus_sparse(W, k1, k2, b=b, desired_sparsity_ratio=0.25)
    # #quantize_LRQ1Q2_plus_sparse(W, k1, k2, b=b)
    # print(f"Error of LR + LR-Quant + Sparse (k1={k1}, k2={k2}, b={b}): {err}")
    # print("\tNumber of bits: %e\n" % num_bits)

    # k = (num_bits - b*W.shape[0]*W.shape[1]) // (32 * (W.shape[0] + W.shape[1]))

    print(f"Testing LQ-LORA + Factors")
    k = 60
    budget = 32*(W.shape[0]+W.shape[1]) * k + b*W.shape[0]*W.shape[1]
    print("\tNumber of bits: %e\n" % budget)
    for k1 in range(k, -1, -10):
        k2 = (budget - 32*(W.shape[0]+W.shape[1]) * k1 - b*W.shape[0]*W.shape[1]) \
                // (b*(W.shape[0]+W.shape[1]))
        err, _ = quantize_LQLORA_split(W, k1, k2, b=b, factor_b=b)
        print(f"Error of LQ-LORA + Factors (k1={k1}, k2={k2}, b={b}): {err}")

    # print(f"Testing LQ-LORA (k={k})...")
    # err, mtxs = quantize_LQLORA(W, k, b=b, n_iter=40)
    # print(f"Error of LQ-LORA (k={k}): {err}")
    # print("\tNumber of bits: %e\n" % (32*(W.shape[0]+W.shape[1]) * k + b*W.shape[0]*W.shape[1]))

    # # # quantize_noncvx(W, 500, 1e-3, b=4, n_iter=50, print_freq=5)

    budget = 32*(W.shape[0]+W.shape[1]) * k + b*W.shape[0]*W.shape[1]

    k1 = k
    b1 = b
    get_k2 = lambda k1, b1, keep_S: (budget - 32*k1*(W.shape[0] + W.shape[1])) \
                // (b1*(W.shape[0] + W.shape[1]) + 32) if keep_S else \
                    (budget - 32*k1*(W.shape[0] + W.shape[1])) // (b1*(W.shape[0] + W.shape[1]))
    k2 = get_k2(k1, b, True)

    print(f"Testing LR^T Q1 Q2^T...")

    while k2 > 0:
        err, mtxs = quantize_LRQ1Q2(W, k1, k2, b=b1, n_iter_outer=5, keep_S=True)
        print(f"k1={k1}, k2={k2}, b={b1}, keep S: {err}")

        k2 = get_k2(k1, b, False)
        err, mtxs = quantize_LRQ1Q2(W, k1, k2, b=b1, n_iter_outer=5)
        print(f"k1={k1}, k2={k2}, b={b1}: {err}")

        k1 += 10
        k2 = get_k2(k1, b, True)


if __name__ == "__main__":
    # N = 500
    # print(f"W = {N} x {N} Gaussian Matrix N(0, 1)")
    # print("-"*80)
    # np.random.seed(10)
    # W = np.random.normal(size=(N, N))
    # test_quant(W)
    # print("-"*80)

    # print("W = AlexNet FC3 (4096 x 1000)")
    # print("-"*80)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # model.eval()
    # W = list(model.classifier[6].parameters())[0].detach().numpy()
    # test_quant(W)
    # print("-"*80)


    print("W = AlexNet FC2 (4096 x 4096)")
    print("-"*80)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()
    W = list(model.classifier[4].parameters())[0].detach().numpy()
    test_quant(W)
    print("-"*80)
