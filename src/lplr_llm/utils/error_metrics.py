from abc import ABC, abstractmethod
import torch

class ErrorMetric(ABC):
    @property
    def name(self):
        return "Error"

    @abstractmethod
    def error(self, X_hat, X_exact=None, relative=True):
        ...

class FroError(ErrorMetric):
    @property
    def name(self):
        return "Frobenius Norm Error"

    def error(self, X_hat, X_exact, relative=True):
            
        err = torch.norm(X_hat - X_exact, p="fro")
        if not relative:
            return err.item()
        return (err / torch.norm(X_exact, p="fro")).item()
    
class SpectralError(ErrorMetric):
    @property
    def name(self):
        return "Spectral Norm Error"

    def error(self, X_hat, X_exact=None, relative=True):
        if X_exact is None:
            X_exact = self.comparison_matrix

        err = torch.linalg.matrix_norm(X_hat - X_exact, ord=2)
        if not relative:
            return err.item()
        return (err / torch.linalg.matrix_norm(X_exact, ord=2)).item()
    
class RandSpectralError(ErrorMetric):
    def __init__(self, oversample=400):
        self.oversample = oversample

    @property
    def name(self):
        return "Randomized Spectral Norm Error"
    
    def error(self, X_hat, X_exact=None, relative=True):
        if X_exact is None:
            X_exact = self.comparison_matrix

        oversample = min(X_hat.shape[0], X_hat.shape[1], self.oversample)

        _, S1, _ = torch.svd_lowrank(X_hat - X_exact, q=oversample)
        _, S2, _ = torch.svd_lowrank(X_exact, q=oversample)

        if not relative:
            return S1[0].item()
        return (S1[0] / S2[0]).item()