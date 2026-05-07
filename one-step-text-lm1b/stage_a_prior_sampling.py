from __future__ import annotations

import torch
import torch.nn.functional as F

from prepare import PAD_TOKEN_ID


def sample_token_ids_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    probs = F.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        remove = (cumprobs - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        batch, seq_len, vocab = sorted_probs.shape
        sampled = torch.multinomial(sorted_probs.reshape(-1, vocab), num_samples=1).reshape(batch, seq_len)
        return sorted_idx.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)

    batch, seq_len, vocab = probs.shape
    return torch.multinomial(probs.reshape(-1, vocab), num_samples=1).reshape(batch, seq_len)


def decode_token_ids(tokenizer, token_ids: torch.Tensor) -> list[str]:
    texts = []
    for row in token_ids.detach().cpu().tolist():
        while row and row[-1] == PAD_TOKEN_ID:
            row.pop()
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def hidden_to_logits(qwen_lm, hidden: torch.Tensor) -> torch.Tensor:
    output_embeddings = qwen_lm.get_output_embeddings()
    hidden = hidden.to(output_embeddings.weight.dtype)
    return output_embeddings(hidden).float()


def summarize_prior_hidden(hidden: torch.Tensor) -> dict[str, float]:
    token_norms = hidden.detach().float().norm(dim=-1)
    hidden = hidden.detach().float()
    return {
        "hidden_mean": hidden.mean().item(),
        "hidden_std": hidden.std().item(),
        "hidden_norm": token_norms.mean().item(),
    }


def summarize_prior_cycle(z: torch.Tensor, z_hat: torch.Tensor) -> dict[str, float]:
    z = z.detach().float()
    z_hat = z_hat.detach().float()
    diff = z_hat - z
    z_flat = z.reshape(z.shape[0], -1)
    z_hat_flat = z_hat.reshape(z_hat.shape[0], -1)
    return {
        "z_cycle_mse": diff.square().mean().item(),
        "z_cycle_mae": diff.abs().mean().item(),
        "z_cycle_cos": F.cosine_similarity(z_flat, z_hat_flat, dim=-1).mean().item(),
    }


@torch.no_grad()
def run_stage_a_prior_sampling(
    stage_a,
    qwen_lm,
    tokenizer,
    device: str,
    num_samples: int,
    seq_len: int,
    z_scale: float,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, float], list[str]]:
    z = torch.randn(num_samples, seq_len, stage_a.d, device=device) * z_scale
    hidden_normalized = stage_a.decode_latents(z, unnormalize=False)
    hidden = stage_a.unnormalize_hidden(hidden_normalized)
    z_hat, _ = stage_a.flow(hidden_normalized)
    logits = hidden_to_logits(qwen_lm, hidden)
    token_ids = sample_token_ids_from_logits(logits, temperature=temperature, top_p=top_p)
    texts = decode_token_ids(tokenizer, token_ids)

    metrics = {}
    metrics.update(summarize_prior_cycle(z, z_hat))
    metrics.update(summarize_prior_hidden(hidden))
    return metrics, texts
