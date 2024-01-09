import math
from typing import Tuple

import numpy as np
import torch
from loguru import logger


def normalize_and_shift_wrt_inner_prod(X: torch.Tensor = None, Y: torch.Tensor = None):
    """
    :param X: Target matrix
    :param Y: Approximation matrix
    :return: Best approximation of X up to scaling and shift of Y
    """

    assert X.shape == Y.shape, "Dimension mismatch!"

    n = X.shape[0]
    d = X.shape[1]
    M = torch.ones_like(X)

    t1 = torch.linalg.norm(Y, ord="fro") ** 2 - (
        torch.sum(torch.multiply(Y, M)) ** 2 / (n * d)
    )
    logger.trace(f"t1 = {t1:.5f}")
    t2 = torch.sum(torch.multiply(X, Y)) - torch.sum(torch.multiply(Y, M)) * torch.sum(
        torch.multiply(X, M)
    ) / (n * d)

    logger.trace(f"t2 = {t2:.5f}")
    alpha = t2 / t1
    logger.trace(f"alpha = {alpha:.5f}")
    beta = (
        torch.sum(torch.multiply(X, M)) - torch.sum(torch.multiply(Y, M)) * alpha
    ) / (n * d)
    logger.trace(f"beta = {beta:.5f}")
    return alpha * Y + beta * M


def sparse_jl_transform(d: int = None, m: int = None, s: int = 1):
    """_summary_
    :param d: Number of rows
    :param m: Number of columns
    :param s: Sparsity level
    Returns a sparse JL transform of size d x m with s non-zero entries in each row
    """

    assert (
        s <= m
    ), "Number of non-zero elements in each row much be less than the number of columns!"

    S = torch.zeros(d, m)

    for i in range(d):
        indices = torch.randperm(m)[:s]
        entries = 2 * torch.randint(0, 2, (s,)) - 1
        for j in range(s):
            S[i][indices[j]] = entries[j]

    return S * 1 / math.sqrt(s)


def maximum_output_rank(
    compression_ratio: float,
    b1: int,
    b2: int,
    b_nq: float,
    input_shape: Tuple[int, int],
):
    numerator = compression_ratio * math.prod(input_shape) * b_nq
    denominator = np.dot(input_shape, (b1, b2))
    return math.floor(numerator / denominator)