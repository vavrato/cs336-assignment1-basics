from typing import Optional
from torch import nn, Tensor
import torch
import math
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int


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


class ROPE(nn.Module):
    # it would be nice to refactor, clean, and make more efficient
    # no time, though, two kids and fulltime job
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.theta_base = theta ** (-2 / d_k)
        t = self.make_angle_tensor(max_seq_len, d_k)

        cos = torch.cos(t)
        sin = torch.sin(t)

        self.register_buffer("cos", cos, persistent=False)  # persistent=False, otherwise problems with load_state_dict
        self.register_buffer("sin", sin, persistent=False)

    def theta_i(self, i):
        return self.theta_base ** (i)

    def make_angle_vector(self, d_k) -> torch.Tensor:
        vector_list = []
        for k in range(0, d_k // 2):
            vector_list.append(self.theta_i(k))

        vector = torch.tensor(vector_list).reshape(1, -1)
        return repeat(vector, "a b -> a (b n)", n=2)

    def make_angle_tensor(self, max_seq_len, d_k) -> torch.Tensor:
        rows = []
        angle_vector = self.make_angle_vector(d_k)
        for m in range(0, max_seq_len):
            rows.append(m * angle_vector)

        return torch.cat(rows)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # first make the rearranged vector that will be multiplied by sin, I don't know if this can be more elegant
        y = x.clone()

        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]

        cos = getattr(self, "cos")[positions, :]
        sin = getattr(self, "sin")[positions, :]
        for _ in range(y.ndim - 3):  # I do not understand at the moment why 3 works, as well as unsqueeze(1). WTF?
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return x * cos + y * sin


def softmax(x: torch.Tensor, *, dim: Optional[int] = None) -> torch.Tensor:
    x_shifted = x - torch.max(x, dim, keepdim=True).values
    exp = torch.exp(x_shifted)
    norm = exp.sum(dim, keepdim=True)

    return exp / norm


def sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Optional[Float[Tensor, " ... queries keys"]] = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    presoftmax = 1 / d_k ** (0.5) * (Q @ rearrange(K, "... keys d_model -> ... d_model keys"))

    if mask is not None:
        presoftmax += torch.where(mask, 0, -torch.inf)

    return softmax(presoftmax, dim=-1) @ V


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        use_rope: bool = False,
        max_seq_len: int = 1,
        theta: float = 10000
    ):
        super().__init__()
        self.num_heads = num_heads
        if not d_k:
            d_k = int(d_model / num_heads)
        if not d_v:
            d_v = int(d_model / num_heads)


        # I think this should be the correct init
        self.WQ = Linear(d_k * num_heads, d_model).W
        nn.init.normal_(self.WQ, 0, 2 / (d_model + d_k))

        self.WK = Linear(d_k * num_heads, d_model).W
        nn.init.normal_(self.WK, 0, 2 / (d_model + d_k))

        self.WV = Linear(d_v * num_heads, d_model).W
        nn.init.normal_(self.WV, 0, 2 / (d_model + d_v))

        self.WO = Linear(d_model, num_heads * d_v).W
        nn.init.normal_(self.WO, 0, 2 / (d_model + d_v))

        self.use_rope = use_rope
        if use_rope:
            self.rope = ROPE(theta, d_k, max_seq_len)

    def forward_suboptimal(self, x: Float[Tensor, "... seq_len d_model"], WQ, WK, WV, WO) -> torch.Tensor:
        # too many matrix multiplications here, move on

        Q = x @ WQ.T  # this is batch, seq_len, n_heads * d_k
        K = x @ WK.T
        V = x @ WV.T

        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)

        seq_len = x.shape[-2]

        mask = rearrange(torch.tril(torch.ones(seq_len, seq_len)).bool(), "s1 s2 -> 1 1 s1 s2")
        attention = sdpa(Q, K, V, mask=mask)  # batch, num_heads, seq_len, d_k

        attention = rearrange(attention, "b h s d -> b s (h d)")  # this is the concatenation

        return attention @ WO.T

    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions=None) -> torch.Tensor:
        # this does just one matrix multiplication
        W = torch.cat(
            [self.WQ, self.WK, self.WV], dim=0
        )  # each W{Q,K,V} is (h*d_k, d_model), so we make (3 * h * d_k, d_model)

        QKV = x @ W.T  # this is ... seq_len 3*h*d_k
        QKV = rearrange(QKV, "... s (three h d_k) -> ... h s (three d_k)", h=self.num_heads, three=3)
        Q, K, V = QKV.chunk(3, dim=-1)  # each is now [... h s d]

        if self.use_rope:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        seq_len = x.shape[-2]
        mask = rearrange(torch.tril(torch.ones(seq_len, seq_len)).bool(), "s1 s2 -> 1 1 s1 s2")
        attention = sdpa(Q, K, V, mask=mask)  # batch, num_heads, seq_len, d_k
        attention = rearrange(attention, "b h s d -> b s (h d)")  # this is the concatenation

        return attention @ self.WO.T

if __name__ == '__main__':
    mha = MultiHeadAttention(8, 4, 2, 2, True, 5)
    mha.forward(torch.randn(3,5,8), token_positions=torch.tensor([0,1,2,3,4]))