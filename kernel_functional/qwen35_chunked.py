# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights
# reserved.
# Adapted from Transformers' chunk_gated_delta_rule, Apache-2.0.
# See ../LICENSE-qwen35-transformers.txt.

import torch
from torch.nn import functional as F

from .error_handling import validate_chunk_inputs


def gated_delta_chunked(
    query, key, value, log_decay, beta, initial_state=None, chunk_size=64
):
    # Compute the same delta recurrence in blocks of tokens.
    # Triangular solves batch the dependencies handled by the sequential token
    # scan.
    # Notation: H=Hv; M=number of chunks; L=chunk size; * means elementwise.
    validate_chunk_inputs(query, chunk_size)
    q, k, v, k_beta, v_beta, cumulative = _prepare_chunks(
        query, key, value, log_decay, beta, chunk_size
    )
    U, W, local_reads = _solve_local_updates(
        q, k, k_beta, v_beta, cumulative, chunk_size
    )
    output, state = _scan_chunks(
        q,
        k,
        v,
        cumulative,
        U,
        W,
        local_reads,
        initial_state,
        query.shape[1],
    )
    # Reads (B,H,T,Dv) -> (B,T,H,Dv); state remains (B,H,Dk,Dv).
    return output.transpose(1, 2).to(query.dtype).contiguous(), state


def _prepare_chunks(query, key, value, log_decay, beta, chunk_size):
    length = query.shape[1]
    # Move heads before time: Q/K (B,T,H,Dk) -> (B,H,T,Dk), V uses Dv.
    # g=log(alpha) and beta -> (B,H,T); all scan arithmetic uses FP32.
    q, k, v, beta, g = (
        x.transpose(1, 2).to(torch.float32).contiguous()
        for x in (query, key, value, beta, log_decay)
    )
    # q_hat = q / sqrt(sum_d q_d^2 + eps); shape stays (B,H,T,Dk).
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    # k_hat = k / sqrt(sum_d k_d^2 + eps); sum is over Dk, not time.
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    # q_read = q_hat / sqrt(Dk), exactly the scaling in the sequential scan.
    q = q * (q.shape[-1] ** -0.5)
    # Pad to T_pad = M*L: beta=0 means no write, g=0 means alpha=1.
    # These artificial positions leave the final memory unchanged.
    padding = (-length) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, padding)) for x in (beta, g))
    # K_beta[t,:] = beta_t * K[t,:]; V_beta[t,:] = beta_t * V[t,:].
    # Beta(B,H,T_pad,1) broadcasts over Dk/Dv.
    k_beta, v_beta = k * beta.unsqueeze(-1), v * beta.unsqueeze(-1)
    # Expose chunks: (B,H,T_pad,D) -> (B,H,M,L,D). No arithmetic here.
    q, k, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (q, k, k_beta, v_beta)
    )
    # G_i = sum_{j=0}^i g_j within EACH chunk; shape (B,H,M,L).
    # exp(G_i) is the cumulative retention from chunk start through token i.
    cumulative = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(-1)

    return q, k, v, k_beta, v_beta, cumulative


def _solve_local_updates(q, k, k_beta, v_beta, cumulative, chunk_size):
    # Decay from position j to i, masked before exp to avoid overflow above
    # diagonal.
    # Mask j>i before exponentiating: only earlier/current writes can affect
    # token i.
    future = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=q.device
    ).triu(1)
    # D_ij = exp(G_i-G_j) for j<=i, else 0; shape (B,H,M,L,L).
    # This retains a write at j through later decays j+1,...,i.
    relative_decay = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    relative_decay = relative_decay.masked_fill(future, -torch.inf).exp()
    # A_ij = beta_i * dot(k_i,k_j) * D_ij.
    # (B,H,M,L,Dk) @ (B,H,M,Dk,L) -> (B,H,M,L,L).
    # The solver uses Lsys = I + tril(A,-1); it ignores A on/above the diagonal.
    system = (k_beta @ k.transpose(-1, -2)) * relative_decay
    # P_ij = dot(q_i,k_j) * D_ij -> (B,H,M,L,L).
    # Unlike the solve system, P includes i=j: a token reads its own write.
    local_reads = (q @ k.transpose(-1, -2)) * relative_decay
    # B_i = beta_i * k_i * exp(G_i) -> (B,H,M,L,Dk).
    # These coefficients describe how incoming memory changes predicted values.
    decayed_writes = k_beta * cumulative.exp().unsqueeze(-1)

    # Diagonal is implicitly 1; only the strict lower triangle is used.
    # Solve Lsys @ U = V_beta: (L,L) @ (L,Dv) = (L,Dv).
    # U accounts for earlier writes inside this chunk, assuming zero incoming
    # memory.
    U = torch.linalg.solve_triangular(
        system, v_beta, upper=False, unitriangular=True
    )
    # Solve Lsys @ W = B: (L,L) @ (L,Dk) = (L,Dk).
    # Given incoming S, actual corrections are U - W @ S.
    W = torch.linalg.solve_triangular(
        system, decayed_writes, upper=False, unitriangular=True
    )
    return U, W, local_reads


def _scan_chunks(
    q,
    k,
    v,
    cumulative,
    U,
    W,
    local_reads,
    initial_state,
    length,
):
    if initial_state is None:
        state = q.new_zeros(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1])
    else:
        state = initial_state.float()
    # Qbar_i = q_i * exp(G_i): incoming S decays before query i reads it.
    q = q * cumulative.exp().unsqueeze(-1)
    # Kbar_i = k_i * exp(G_last-G_i): each write decays until chunk end.
    k = k * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    # alpha_chunk = exp(G_last), shape (B,H,M,1,1), scales the incoming S.
    final_decay = cumulative[..., -1].exp().unsqueeze(-1).unsqueeze(-1)
    outputs = []
    for i in range(q.shape[2]):
        # Delta = U - W @ S: (L,Dv) - (L,Dk) @ (Dk,Dv) -> (L,Dv).
        # All equations in this loop also carry batch/head axes (B,H).
        corrections = U[:, :, i] - W[:, :, i] @ state
        # Y = Qbar @ S + P @ Delta: (L,Dk)@(Dk,Dv) + (L,L)@(L,Dv).
        # One (B,H,L,Dv) block combines old memory and this chunk's new writes.
        outputs.append(q[:, :, i] @ state + local_reads[:, :, i] @ corrections)
        # S_next = alpha_chunk*S + Kbar.T @ Delta.
        # (Dk,L) @ (L,Dv) -> (Dk,Dv); only this state crosses chunk boundaries.
        state = (
            state * final_decay[:, :, i]
            + k[:, :, i].transpose(-1, -2) @ corrections
        )
    # Stack (B,H,M,L,Dv) -> flatten (B,H,M*L,Dv) -> remove padded tokens.
    # The public wrapper returns (B,T,H,Dv) in the original query dtype.
    output = torch.stack(outputs, dim=2).flatten(2, 3)[:, :, :length]
    return output, state
