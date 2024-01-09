import numpy as np
from scipy import stats
import torch
from sklearn.utils.extmath import randomized_svd as rand_svd
import matplotlib.pyplot as plt
from quant_helpers import *

# svd = np.linalg.svd
svd = rand_svd

def quantize_cvx(W, k_max, lmbda, b=4, print_freq=1, n_iter=10, rtol=1e-6):
    one_coeff = 10
    mu = W.shape[0] * W.shape[1] / (4 * np.sum(np.abs(W)))
    L = W

    U, S, VT = svd(W, k_max)
    S = np.diag(S)
    L = U @ S @ VT

    Q_max = np.max(np.abs(W-L))
    Q = nf_quant(W - L, b=b, A_max=Q_max)

    q = nf_values(Q_max, b=b)

    Qi = []
    for i in range(2**b):
        Qi.append(np.abs(Q - q[i]) < Q_max*rtol)
        # Qi.append(np.ones_like(W))
    
    Q_quant = nf_quant(Q, b=b, A_max=Q_max)
    print(f"\t[DEBUG] Initial Error: {np.linalg.norm(W - L - Q, 'fro')}")
    print(f"\t[DEBUG] Initial Rank: {k_max}")

    one = np.ones_like(W)

    for j in range(n_iter):
        if j % print_freq == 0:
            print(f"\t[DEBUG] {j}")

        ## Set L
        # U, S, VT = svd(W - Q - 1/mu * Y1, k_max)
        U, S, VT = svd(W - Q, k_max)
        S = np.diag(shrinkage(mu, S))
        L = U @ S @ VT

        rank = np.sum(S > 0)

        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set L): {np.linalg.norm(W - L - Q, 'fro')}")
            # print(f"\t\t[DEBUG] Quant Error (set L): {np.linalg.norm(W - L - Q_quant, 'fro')}")
            print(f"\t\t[DEBUG] Rank: {rank}\n")

        ## Set Q
        for i in range(2**b):
            Qi[i] = 1/((q[i]**2 + one_coeff) * mu) * shrinkage(
                lmbda,
                mu * (q[i] * (W - L - Q + q[i] * Qi[i]) + one_coeff*(one - sum(Qi) + Qi[i]))
                # q[i] * Y1 + mu * (q[i] * (W - L - Q + q[i] * Qi[i]))
            )
            Q = sum([q[i] * Qi[i] for i in range(2**b)])

        Q_quant = nf_quant(Q, b=b, A_max=Q_max)
        # Q = Q_quant
        # Qi = []
        # for i in range(2**b):
        #     Qi.append(1 * (np.abs(Q - q[i]) < rtol*Q_max))

        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set Q): {np.linalg.norm(W - L - Q, 'fro')}")
            # print(f"\t\t[DEBUG] ||sum qi - 1||: {np.linalg.norm(sum(Qi) - one, 'fro')}")
            print(f"\t\t[DEBUG] Quant Error: {np.linalg.norm(W - L - Q_quant, 'fro')}")

        ## Set Y
        # Y1 += mu * (W - L - Q)

    plt.hist([Qii.flatten() for Qii in Qi], bins=10)
    # plt.hist(Q.flatten(), bins=50)
    plt.show()

def quantize_noncvx(W, k_max, lmbda, b=4, print_freq=1, n_iter=10, rtol=1e-6):
    one_coeff = 1
    mu = W.shape[0] * W.shape[1] / (4 * np.sum(np.abs(W)))
    U, S, VT = svd(W, k_max)
    S = np.diag(S)
    L = U @ S @ VT

    Q_max = np.max(np.abs(W-L))
    Q = nf_quant(W - L, b=b, A_max=Q_max)

    q = nf_values(Q_max, b=b)

    Qi = []
    for i in range(2**b):
        Qi.append(np.abs(Q - q[i]) < Q_max*rtol)
        # Qi.append(np.ones_like(W))
    
    Q_quant = nf_quant(Q, b=b, A_max=Q_max)
    print(f"\t[DEBUG] Initial Error: {np.linalg.norm(W - L - Q, 'fro')}")
    print(f"\t[DEBUG] Initial Rank: {k_max}")

    one = np.ones_like(W)

    plt.hist([Qii.flatten() for Qii in Qi], bins=10)
    # plt.hist(Q.flatten(), bins=50)
    plt.show()

    lmbda_step = 5e-3
    one_coeff_step = 0.5

    for j in range(n_iter):
        if j % print_freq == 0:
            print(f"\t[DEBUG] {j}")

        ## Set L
        U, S, VT = svd(W - Q, k_max)
        S = np.diag(shrinkage(mu, S))
        L = U @ S @ VT

        # Q_max = np.max(np.abs(W-L))
        # q = nf_values(Q_max, b=b)
        # Q = sum([q[i] * Qi[i] for i in range(2**b)])

        rank = np.sum(S > 0)

        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set L): {np.linalg.norm(W - L - Q, 'fro')}")
            # print(f"\t\t[DEBUG] Quant Error (set L): {np.linalg.norm(W - L - Q_quant, 'fro')}")
            print(f"\t\t[DEBUG] Rank: {rank}\n")

        ## Set Q
        
        for i in range(2**b):
            alpha = (q[i]**2 + one_coeff) * mu
            beta = mu * (q[i] * (W - L - Q + q[i] * Qi[i]) + one_coeff*(one - sum(Qi) + Qi[i]))
            # beta = q[i] * Y1 + mu * (q[i] * (W - L - Q + q[i] * Qi[i]))

            obj_contrib_fn = lambda Qii: mu * (W - L - Q + q[i] * Qi[i] - q[i] * Qii) ** 2 + \
                                  lmbda * np.minimum(np.abs(Qii), np.abs(Qii - 1)) + \
                                  one_coeff * mu * (one - sum(Qi) + Qi[i] - Qii) ** 2

            Qii_best = Qi[i]
            obj_contrib_best = obj_contrib_fn(Qi[i])

            ## < 0 or between 1/2 and 1
            Qii_1 = np.maximum(0, np.minimum(1, 1/alpha * (beta + lmbda)))
            obj_contrib_1 = obj_contrib_fn(Qii_1)

            Qii_best = (obj_contrib_1 < obj_contrib_best) * Qii_1 + \
                        (obj_contrib_1 >= obj_contrib_best) * Qii_best
            obj_contrib_best = (obj_contrib_1 < obj_contrib_best) * obj_contrib_1 + \
                        (obj_contrib_1 >= obj_contrib_best) * obj_contrib_best

            ## > 1 or between 0 and 1/2
            Qii_2 = np.maximum(0, np.minimum(1, 1/alpha * (beta - lmbda)))
            obj_contrib_2 = obj_contrib_fn(Qii_2)

            Qii_best = (obj_contrib_2 < obj_contrib_best) * Qii_2 + \
                        (obj_contrib_2 >= obj_contrib_best) * Qii_best
            obj_contrib_best = (obj_contrib_2 < obj_contrib_best) * obj_contrib_2 + \
                        (obj_contrib_2 >= obj_contrib_best) * obj_contrib_best

            ## = 0
            Qii_3 = np.zeros_like(W)
            obj_contrib_3 = obj_contrib_fn(Qii_3)

            Qii_best = (obj_contrib_3 < obj_contrib_best) * Qii_3 + \
                        (obj_contrib_3 >= obj_contrib_best) * Qii_best
            obj_contrib_best = (obj_contrib_3 < obj_contrib_best) * obj_contrib_3 + \
                        (obj_contrib_3 >= obj_contrib_best) * obj_contrib_best

            ## = 1
            Qii_4 = np.ones_like(W)
            obj_contrib_4 = obj_contrib_fn(Qii_4)
            Qii_best = (obj_contrib_4 < obj_contrib_best) * Qii_4 + \
                        (obj_contrib_4 >= obj_contrib_best) * Qii_best
            obj_contrib_best = (obj_contrib_4 < obj_contrib_best) * obj_contrib_4 + \
                        (obj_contrib_4 >= obj_contrib_best) * obj_contrib_best
            
            Qi[i] = Qii_best

            Q = sum([q[i] * Qi[i] for i in range(2**b)])

        Q_quant = nf_quant(Q, b=b, A_max=Q_max)
        
        if j % print_freq == 0:
            print(f"\t\t[DEBUG] Error (set Q): {np.linalg.norm(W - L - Q, 'fro')}")
            print(f"\t\t[DEBUG] Quant Error (set Q): {np.linalg.norm(W - L - Q_quant, 'fro')}")

        ## Set Y
        # Y1 += mu * (W - L - Q)
        # Y2 += mu * (one - sum(Qi))
        one_coeff += one_coeff_step
        lmbda *= 1.1


    plt.hist([Qii.flatten() for Qii in Qi], bins=10)
    # plt.hist(Q.flatten(), bins=50)
    plt.show()

