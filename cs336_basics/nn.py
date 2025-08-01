from typing import Optional
from torch import nn
import torch
import math
from einops import rearrange, reduce, repeat


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
        device: torch.device | None = None, # type: ignore
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

class ROPE(nn.Module):
    # it would be nice to refactor, clean, and make more efficient
    # no time, though, two kids and fulltime job
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device]=None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.theta_base = theta**(-2/d_k)
        t = self.make_angle_tensor(max_seq_len, d_k)

        cos = torch.cos(t)
        sin = torch.sin(t)

        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def theta_i(self, i):
        return self.theta_base**(i)

    def make_angle_vector(self, d_k) -> torch.Tensor:
        vector_list = []
        for k in range(0,d_k//2):
            vector_list.append(self.theta_i(k))
            
        vector = torch.tensor(vector_list).reshape(1,-1)
        return repeat(vector, 'a b -> a (b n)', n=2)
    
    def make_angle_tensor(self, max_seq_len, d_k) -> torch.Tensor:
        rows = []
        angle_vector = self.make_angle_vector(d_k)
        for m in range(0,max_seq_len):
            rows.append(m * angle_vector)

        return torch.cat(rows)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # first make the rearranged vector that will be multiplied by sin, I don't know if this can be more elegant
        y = x.clone()

        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]

        cos_tensor = getattr(self, "cos")
        sin = getattr(self, "sin")
        return x * cos_tensor[positions, :] + y * sin[positions, :]

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    x_shifted = x - torch.max(x, dim).values
    exp = torch.exp(x_shifted)
    norm = exp.sum(dim)

    return exp/norm