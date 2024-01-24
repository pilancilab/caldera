import torch
from loguru import logger
from lplr_llm.quantization import QuantizerFactory, quantize_small_sv_components, mixed_precision_quantize
from tqdm import tqdm
from lplr_llm.weight_compressors import alternating_mixed_lplr

def make_sparse(A, sparsity):
    A_flat = torch.abs(A.flatten())
    nth_largest = torch.topk(A_flat, sparsity).values[-1]
    return A * (torch.abs(A) > nth_largest)

def weight_decomposition_admm(
    X: torch.Tensor,
    rank: int = None,
    sparsity: int = None,
    rho_admm: float = None,
    BQ: int = 4,
    BLR: int = 16,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
    iters: int = 50,
    log_errors: bool = False,
    verbose: bool = False
):
    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"

    ###########################################################################
    ## Optimization variables:
    ## - Q: quantized matrix
    ## - Q': real-valued matrix, constrained to be = Q
    ## - L, R: rank-m matrices such that LR is the size of X
    ## - S: k-sparse matrix
    ###########################################################################
    ## Optimization problem:
    ## min 1/2 ||X - LR - Q - S||^2_F s.t. Q=Q'
    ###########################################################################
    ## Initialization: similar to LoftQ
    ## - Q = X, Q' = Quant(X)
    ## - L, R via SVD
    ## - S = 0
    ###########################################################################
    quant_Q = quantizer_factory.get_quantizer(BQ, device=X.device)
    quant_LR = quantizer_factory.get_quantizer(BLR, device=X.device)

    # n, d = X.shape

    Q = quant_Q.dequantize_block(*quant_Q.quantize_block(X))
    Qp = Q

    U, Sigma, VT = torch.linalg.svd(X - Qp)
    sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:rank]))
    L = U[:, :rank] @ sqrt_Sigma
    R = sqrt_Sigma @ VT[:rank, :]

    S = torch.zeros_like(X).to(X.device)
    lmbda = torch.zeros_like(X).to(X.device)

    errors = []
    errors_Qp = []
    lagrangians = []
    constraint_vals = []

    for _ in tqdm(range(iters)):
        ## Update L, R
        # if BLR >= 16: # full-precision 
        #     U, Sigma, VT = torch.linalg.svd(X - Qp - S)
        #     sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:rank]))
        #     L = U[:, :rank] @ sqrt_Sigma
        #     R = sqrt_Sigma @ VT[:rank, :]
        # else: # quantized, so use least squares

        # ## Update S
        if sparsity > 0:
            S = make_sparse(X - L @ R - Qp, sparsity)

        Y = torch.linalg.lstsq(R.T, (X - Qp - S).T)[0].T
        # Y = (X - Qp - S) @ torch.linalg.pinv(R)
        if not torch.isnan(Y).any().item():
            L = quant_LR.dequantize_block(*quant_LR.quantize_block(Y))
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized L.")
        # U, Sigma, VT = torch.linalg.svd(X - Qp - S)
        # sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:rank]))
        # L = quant_LR.dequantize_block(*quant_LR.quantize_block(U[:, :rank] @ sqrt_Sigma))

        W = torch.linalg.lstsq(L, X - Qp - S)[0]
        # W = torch.linalg.pinv(L) @ (X - Qp - S)
        if not torch.isnan(W).any().item():
            R = quant_LR.dequantize_block(*quant_LR.quantize_block(W))
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized R.")

        ## Update Q'
        Qp = 1/(1 + rho_admm) * (X - L @ R - S - lmbda + rho_admm * Q)

        ## Update Q
        Q = quant_Q.dequantize_block(*quant_Q.quantize_block(lmbda / rho_admm + Qp))

        ## Update lambda
        lmbda = lmbda + rho_admm * (Qp - Q)

        lagrangian = 1/2 * torch.norm(X - L @ R - Qp - S, p="fro").item() ** 2 + \
            torch.trace(lmbda.T @ (Qp - Q)).item() + rho_admm/2 * \
                torch.norm(Qp - Q, p="fro").item() ** 2
        error = torch.norm(X - L @ R - Q - S, p="fro").item() \
            / torch.norm(X, p="fro").item()
        error_Qp = torch.norm(X - L @ R - Qp - S, p="fro").item() \
            / torch.norm(X, p="fro").item()
        
        constraint_vals.append(torch.norm(Q - Qp, p="fro").item())                
        lagrangians.append(lagrangian)
        errors.append(error)
        errors_Qp.append(error_Qp)
    out = L @ R + Q + S

    if log_errors:
        return (L, R, Q, S), out, (errors, errors_Qp, lagrangians, constraint_vals)
    return (L, R, Q, S), out
        
def weight_decomposition_no_admm(
    X: torch.Tensor,
    ranks: list[int] = None,
    sparsity: int = None,
    BQ: int = 4,
    BLR_list: list[int] = [16],
    BS: int = 16,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
    iters: int = 50,
    log_errors: bool = False,
    verbose: bool = True
):
    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"


    if BQ > 0:
        quant_Q = quantizer_factory.get_quantizer(BQ, device=X.device)
    quant_LRs = [
        quantizer_factory.get_quantizer(B, device=X.device) \
            for B in BLR_list
    ]
    quant_S = quantizer_factory.get_quantizer(BS, device=X.device)

    if BQ > 0:
        Q = quant_Q.dequantize_block(*quant_Q.quantize_block(X))
    else:
        Q = torch.zeros_like(X)

    k = sum(ranks)
    U, Sigma, VT = torch.linalg.svd(X - Q)
    sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:k]))
    L = mixed_precision_quantize(U[:,:k] @ sqrt_Sigma, ranks, quant_LRs)
    R = mixed_precision_quantize(VT.T[:, :k] @ sqrt_Sigma, ranks, quant_LRs).T

    S = torch.zeros_like(X).to(X.device)

    errors = []

    for _ in tqdm(range(iters)):
        ## Update Q
        if BQ > 0:
            Q = quant_Q.dequantize_block(*quant_Q.quantize_block(X - L @ R - S))

        # ## Update S
        if sparsity > 0:
            S = make_sparse(X - L @ R - Q, sparsity)
            if BS < 16:
                S = quant_S.dequantize_block(*quant_S.quantize_block(S))

        ## Update L, R
        Y = torch.linalg.lstsq(R.T, (X - Q - S).T)[0].T
        # Y = (X - Q - S) @ torch.linalg.pinv(R)
        if not torch.isnan(Y).any().item():
            L = mixed_precision_quantize(Y, ranks, quant_LRs)
            # L = quant_LR.dequantize_block(*quant_LR.quantize_block(Y))
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized L.")
        # U, Sigma, VT = torch.linalg.svd(X - Qp - S)
        # sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:rank]))
        # L = quant_LR.dequantize_block(*quant_LR.quantize_block(U[:, :rank] @ sqrt_Sigma))

        W = torch.linalg.lstsq(L, X - Q - S)[0]
        # W = torch.linalg.pinv(L) @ (X - Q - S)
        if not torch.isnan(W).any().item():
            R = mixed_precision_quantize(W.T, ranks, quant_LRs).T   
            # R = quant_LR.dequantize_block(*quant_LR.quantize_block(W))
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized R.")

        error = torch.norm(X - L @ R - Q - S, p="fro").item() \
            / torch.norm(X, p="fro").item()
        
        errors.append(error)
    out = L @ R + Q + S

    if log_errors:
        return (L, R, Q, S), out, (errors)
    return (L, R, Q, S), out
        