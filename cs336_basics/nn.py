from typing import Optional
from torch import nn
import torch
import math
from einops import rearrange, reduce


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.in_features = in_features

        var = 2 / (in_features + out_features)
        sigma = math.sqrt(var)

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, 0, 1, a=-3, b=3)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.W)[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        RMS = reduce(x**2, "... d -> ... 1", "mean")
        RMS = torch.sqrt(RMS + self.eps)

        x = (x / RMS) * self.W

        return x.to(in_dtype)


class Silu(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


def silu(x: torch.Tensor):
    return x * nn.functional.sigmoid(x)


class FFN_SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ):
        super().__init__()
        if not d_ff:
            d_ff = int(d_model * 8 / 3)

        var_13 = 2 / (d_model + d_ff)
        sigma = math.sqrt(var_13)

        W1 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        W3 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        W1 = nn.init.trunc_normal_(W1, 0, sigma, a=-3 * sigma, b=3 * sigma)
        W3 = nn.init.trunc_normal_(W3, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W1 = nn.Parameter(W1)
        self.W3 = nn.Parameter(W3)

        W2 = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        W2 = nn.init.trunc_normal_(W2, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W2 = nn.Parameter(W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = silu(x @ self.W1.T)
        b = x @ self.W3.T

        return (a * b) @ self.W2.T


def ffn_swiglu(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor) -> torch.Tensor:
    a = silu(x @ W1.T)
    b = x @ W3.T

    return (a * b) @ W2.T
