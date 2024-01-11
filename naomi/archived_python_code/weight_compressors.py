def alternating_mixed_lplr_plus_q(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    BQ: int = 4,
    quant_type:int = QuantType.UNIFORM,,
    normalize_and_shift=False,
    outer_iters=10,
    inner_iters=10,
    log_errors=False
):
    """
    Replace the low-rank decomposition step of LoftQ with alternating mixed LPLR
    (see alternating_mixed_lplr). In the end, we obtain an approximation of X by
    the sum of a low-precision low-rank factorization and a full-rank
    lower-precision matrix.

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors
        will be dropped.
    r1 (int, optional): No. of singular vectors to be kept in full precision
        for the first factor; Trailing (k - r) singular vectors will be quantized.
    r2 (int, optional): No. of singular vectors to be kept in full precision
        for the second factor.
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional): Bit-budget for second low-rank factor
    BQ (int, optional): Bit-budget for the full-rank matrix. This can be lower
        than B1, B2 while retaining good approximation accuracy.
    quantization_fn (function, optional): either `quantize` or `quantize_nf`;
        specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for better
        approximation.
    iters (int, optional): Number of iterations of alternating optimization
    [TODO] sketch (Sketch, optional): Sketch type
    log_errors (bool, optional): Return fro-norm errors for each iteration

    Outputs:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor] : (L, R, Q) such that X
        is approximated by Q + LR.
    - torch.Tensor: Approximation of X (Q + LR)
    - [only if log_errors] list[float]: fro-norm errors for each iteration.
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    L = torch.zeros(X.shape[0], k).to(X.device)
    R = torch.zeros(k, X.shape[1]).to(X.device)

    best_error = float('inf')
    best_mtxs = None
    errors = []

    for _ in range(1, outer_iters):
        Q = quantization_fn(X - L @ R, BQ)

        error = torch.norm(X - L @ R - Q, p="fro").item()
        if error < best_error:
            best_error = error
            best_mtxs = (L, R, Q)

        mtxs, LR = alternating_mixed_lplr(
            X=X-Q,
            k=k, r1=r1, r2=r2,
            B1=B1, B2=B2,
            quantization_fn=quantization_fn,
            normalize_and_shift=normalize_and_shift,
            iters=inner_iters
        )
        error = torch.norm(X - LR - Q, p="fro").item()
        L, R = mtxs

        errors.append(error)
        if error < best_error:
            best_error = error
            best_mtxs = (L, R, Q)

    L, R, Q = best_mtxs
    out = L @ R + Q
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R + Q)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    if log_errors:
        return (L, R, Q), out, errors

    return (L, R, Q), out

def direct_svd_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quantization_fn=quantize,
    normalize_and_shift=False,
):
    """
    Obtain a low-precision low-rank approximation of X without alternating
    optimization by directly quantizing the factorization produced by taking
    the SVD of X.

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors
        will be dropped.
    r1 (int, optional): No. of singular vectors to be kept in full precision
        for the first factor; Trailing (k - r) singular vectors will be quantized.
    r2 (int, optional): No. of singular vectors to be kept in full precision
        for the second factor.
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional):Bit-budget for second low-rank factor
    quantization_fn (function, optional): either `quantize` or `quantize_nf`;
        specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for
        better approximation.
    [TODO] sketch (Sketch, optional): Sketch type

    Outputs:
    - Tuple[torch.Tensor, torch.Tensor] : Low rank quantized factors (L, R)
    - torch.Tensor: Approximation of X (LR)
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

    U = U[:, 0:k]
    S = S[0:k]
    VT = VT[0:k, :]
    S_sqrt = torch.diag(torch.sqrt(S))

    # Get the initial left low-rank factor
    L = quantize_small_sv_components(U @ S_sqrt, B1, r1, quantization_fn=quantization_fn)
    if torch.isnan(L).any().item():
        logger.error(f"NaNs encountered in quantizing first factor")

    R = quantize_small_sv_components(S_sqrt @ VT, B2, r2, quantization_fn=quantization_fn)
    if torch.isnan(R).any().item():
        logger.error(f"NaNs encountered in quantizing second factor")

    out = L @ R
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    return (L, R), out

def loftq(
    X: torch.Tensor = None,
    r: int = None,
    B: int = 8,
    quantization_fn=quantize_nf,
    normalize_and_shift=False,
    iters=10,
    log_errors=False
):
    """
    Uses LoftQ (aka, LQ-LORA) to obtain an approximation of X by the sum of a
    full-precision low-rank factorization and a full-rank low-precision matrix,
    i.e., X_hat = LR + Q.

    This is achieved via alternating optimization: fixing LR, quantize
    (X_hat - LR) to get Q. Then, fixing Q, take a truncated SVD of (X_hat - Q)
    to get L, R.

    X (torch.Tensor, optional): Input matrix (Tall or square)
    r (int, optional): Dimension of the low-rank factor.
    B (int, optional): Bit-budget for the quantized part.
    quantization_fn (function, optional): either `quantize` or `quantize_nf`;
        specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for
        better approximation.
    iters (int, optional): Number of iterations of alternating optimization.
    log_errors (bool, optional): Return fro-norm errors for each iteration.

    Outputs:
    - Tuple[torch.Tensor, torch.Tensor] : Low rank quantized factors (L, R, Q)
    - torch.Tensor: Approximation of X (LR + Q)
    - [only if log_errors] list[float]: fro-norm errors for each iteration.
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"

    n, d = X.shape
    L = torch.zeros(n, r).to(X.device)
    R = torch.zeros(r, d).to(X.device)

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in tqdm(range(iters)):
        Q = quantization_fn(X - L @ R, B)
        
        # Compute full SVD
        U, S, VT = torch.linalg.svd((X - Q).float(), full_matrices=False)

        U = U[:, 0:r]
        S = S[0:r]
        VT = VT[0:r, :]
        S_sqrt = torch.diag(torch.sqrt(S))
        
        L = U @ S_sqrt
        R = S_sqrt @ VT

        errors.append(torch.norm(X - L @ R - Q, p="fro").item())
        if errors[-1] < best_error:
            best_error = errors[-1]
            best_mtxs = (L, R, Q)        
    L, R, Q = best_mtxs
    out = L @ R + Q
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R + Q)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in loftq-ed matrix")

    if log_errors:
        return (L, R, Q), out, errors

    return (L, R, Q), out