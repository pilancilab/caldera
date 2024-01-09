import torch
import numpy as np
from scipy import stats

def quantize(
    X: torch.Tensor = None,
    B: int = 16,
) -> torch.Tensor:
    """Element-wise matrix quantization for general bit-budget
    :param X (torch.Tensor): Matrix to be quantized
    :param B (int): Bit-budget per coordinate for quantization
    :return: Quantized matrix
    """

    M = 2**B  # No. of quantization points per dimension
    res = 2 / (M - 1)  # Resolution

    # Normalize the coordinates of the quantizer input to [-1,1]
    X_min = X.min().item()
    X_max = X.max().item()

    X = torch.from_numpy(np.interp(X.to("cpu").numpy(), (X_min, X_max), (-1, 1))).to(
        X.device
    )

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    fst_pt = -1  # First quantization point
    L_idx = torch.floor((X - fst_pt) / res)  # Lower index for each entry
    L = fst_pt + L_idx * res  # Matrix of lower quantization points
    U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

    # Nearest neighbor quantization
    Q = torch.zeros_like(X)
    Q[X < -1] = -1  # Value less than lower limit of dynamic range
    Q[X > 1] = 1  # Value more than upper limit of dynamic range
    mask0 = torch.abs(X) <= 1  # Value within dynamic range
    mask = torch.abs(X - L) <= res / 2
    Q[mask * mask0] = L[mask * mask0]
    mask = torch.abs(U - X) <= res / 2
    Q[mask * mask0] = U[mask * mask0]

    # Re-normalize the quantized matrix back to its input scale
    Qr = torch.from_numpy(
        np.interp(
            Q.to("cpu").numpy(),
            (Q.min().item(), Q.max().item()),
            (X_min, X_max),
        )
    ).to(X.device)

    return Qr

def quantize_nf(
    X: torch.Tensor = None,
    b: int = 16 
) -> torch.Tensor:
    """
    Element-wise normal float quantization (as introduced by QLORA)

    :param X (torch.Tensor): Matrix to be quantized
    :param B (int): Bit-budget per coordinate for quantization
    :return: Quantized matrix
    """
    
    # Normalize the coordinates of the quantizer input to [-1,1]
    X_min = X.min().item()
    X_max = X.max().item()

    X = torch.from_numpy(np.interp(X.to("cpu").numpy(), (X_min, X_max), (-1, 1))).to(
        X.device
    )

    # We quantize the range [-1, 0) and the range [0, 1] separately, with
    # each having 2^{b-1} levels.
    #
    # The quantization levels are found as follows: take 2^{b-1} evenly-spaced
    # points from [delta, 1/2] and 2{b-1} + 1 from [1/2, 1-delta], where delta
    # is as defined below. The quantization levels are the corresponding
    # quantiles of a standard normal distribution, scaled such that they lie
    # in the range [-1, 1].
    M = 2**(b-1)
    delta = 1/2 * (1/30 + 1/32) # as described above
    res_neg = (1/2 - delta) / (M - 1) # resolution for [delta, 1/2]
    res_pos = (1/2 - delta) / M # resolution for [1/2, 1-delta]
    scale_factor = 1 / stats.norm.ppf(1-delta) # scales the quantization
                                               # levels to be in [-1, 1]

    # quantization levels for the negative and positive halves, respectively
    q_neg = stats.norm.ppf(res_neg * np.arange(M) + delta) * scale_factor
    q_pos = stats.norm.ppf(res_pos * np.arange(M + 1) + 1/2) * scale_factor
    
    # We index into q_neg and q_pos with these indices to get the quantized
    # values for the negative and positive parts of A, respectively.
    neg_quant_idxs = (X < 0) * np.round((stats.norm.cdf(X * stats.norm.ppf(1-delta)) - delta) / res_neg).astype(int)
    pos_quant_idxs = (X >= 0) * np.round((stats.norm.cdf(X * stats.norm.ppf(1-delta)) - 1/2) / res_pos).astype(int)

    Q = (X < 0) * q_neg[neg_quant_idxs] + (X >= 0) * q_pos[pos_quant_idxs]
    
    # Re-normalize the quantized matrix back to its input scale
    Qr = torch.from_numpy(
        np.interp(
            Q.to("cpu").numpy(),
            (Q.min().item(), Q.max().item()),
            (X_min, X_max),
        )
    ).to(X.device)

    return Qr

def quantize_small_sv_components(
    X: torch.Tensor = None,
    B: int = 16,
    r=None,
    quantization_fn=quantize
) -> torch.Tensor:
    """
    Keep the first r columns in original dtype and quantize the last (X.shape[1] - r) columns
    The parameter `quantization_fn` allows you to specify regular quantization (via the
    `quantize` function) or normal float quantization (via the `quantize_nf` function)
    """

    assert r <= X.shape[1], "r should be less than X.shape[1]"
    return torch.cat((X[:, :r], quantization_fn(X[:, r:], B)), dim=1)