from abc import ABC, abstractmethod
import torch

class ErrorMetric(ABC):
    @property
    def name(self):
        return "Error"

    @abstractmethod
    def error(self, X_hat, X_exact=None):
        ...

class FroError(ErrorMetric):
    @property
    def name(self):
        return "Frobenius Norm Error"

    def error(self, X_hat, X_exact):
            
        return (torch.norm(X_hat - X_exact, p="fro") / \
                torch.norm(X_exact, p="fro")).item()
    
class SpectralError(ErrorMetric):
    @property
    def name(self):
        return "Spectral Norm Error"

    def error(self, X_hat, X_exact=None):
        if X_exact is None:
            X_exact = self.comparison_matrix
        return (torch.linalg.matrix_norm(X_hat - X_exact, ord=2) / \
                torch.linalg.matrix_norm(X_exact, ord=2)).item()
    
class RandSpectralError(ErrorMetric):
    def __init__(self, oversample=400):
        self.oversample = oversample

    @property
    def name(self):
        return "Randomized Spectral Norm Error"
    
    def error(self, X_hat, X_exact=None):
        if X_exact is None:
            X_exact = self.comparison_matrix

        oversample = min(X_hat.shape[0], X_hat.shape[1], self.oversample)

        _, S1, _ = torch.svd_lowrank(X_hat - X_exact, q=oversample)
        _, S2, _ = torch.svd_lowrank(X_exact, q=oversample)

        return (S1[0] / S2[0]).item()