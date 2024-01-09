def quantize_LRQ1Q2_plus_sparse_no_rpca(W, k1, k2, sparsity_ratio=0.3, b=4, print_freq=5, epsilon=1e-8, n_iter_outer=20, stop_tolerance=1e-3):
    # print("Running RPCA...")
    # LR, S = inexact_RPCA(W, k1+k2, lmbda)
    # print("RPCA done.")
    U, Sigma, VT = svd(W, k1+k2)
    Sigma = np.diag(Sigma)
    V = VT.T

    L = U[:, :k1] @ np.sqrt(Sigma[:k1, :k1])
    R = V[:, :k1] @ np.sqrt(Sigma[:k1, :k1])

    Q1 = nf_quant_blocked(U[:, k1:] @ np.sqrt(Sigma[k1:, k1:]), b=b)
    Q2 = nf_quant_blocked(V[:, k1:] @ np.sqrt(Sigma[k1:, k1:]), b=b)

    sparsity = int(np.floor(sparsity_ratio*W.shape[1]*W.shape[0]))
    S = nf_quant(make_sparse(W - L @ R.T - Q1 @ Q2.T, sparsity), b=b)

    print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (init): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

    for i in range(n_iter_outer):
        err_before = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')
        if i % print_freq == 0:
            print(f"\t[DEBUG] {i}")
        ## set S
        S = nf_quant(make_sparse(W - L @ R.T - Q1 @ Q2.T, sparsity), b=b)
        if i % print_freq == 0:
            print(f"\t\t[DEBUG] Frobenius norm of W - W_hat (set S): {np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')}")

        ## set Q1, Q2
        for _ in range(10):
            err1 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')
            Q1_old, Q2_old = np.copy(Q1), np.copy(Q2)

            Q1 = nf_quant_blocked(np.linalg.lstsq(Q2, W.T - R @ L.T - S.T, rcond=None)[0].T, b=b)
            Q2 = nf_quant_blocked(np.linalg.lstsq(Q1, W - L @ R.T - S, rcond=None)[0].T, b=b)

            err2 = np.linalg.norm(W - L @ R.T - Q1 @ Q2.T - S, 'fro')

            if err1 < err2:
                Q1, Q2 = Q1_old, Q2_old # restore the old values if this iteration made things worse

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