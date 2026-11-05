"""Metrics, the mutation score and the AnCor losses."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def ndcg(y_true, y_score, top=10):
    """NDCG@top. Adapted from ProteinGym (MIT License): https://github.com/OATML-Markslab/ProteinGym"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    gains = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
    ranks = np.argsort(np.argsort(-y_score)) + 1

    ranks_k = ranks[ranks <= top]
    gains_k = gains[ranks <= top]
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]
    if len(ranks_fil) == 0:
        return 0
    dcg = np.sum([g / np.log2(r + 1) for r, g in zip(ranks_fil, gains_fil)])

    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= top]
    ideal_gains_k = gains[ideal_ranks <= top]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum([g / np.log2(r + 1) for r, g in zip(ideal_ranks_fil, ideal_gains_fil)])
    return dcg / idcg


def mask_marginal(model, seq, mask, wt, pos, tokenizer):
    """Masked-marginal score of each variant against the wild type, and the logits."""
    mask_seq = seq.clone()
    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mask_seq[i, pos[i] + 1] = tokenizer.mask_token_id

    logits = model(input_ids=mask_seq, attention_mask=mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size).to(seq.device)
    for i in range(batch_size):
        site = pos[i] + 1
        scores[i] = torch.sum(log_probs[i][site, seq[i][site]]) - torch.sum(log_probs[i][site, wt[i][site]])
    return scores, logits


def fitness_scale(fitness, rel_tol=1e-8):
    """Standard deviation of the training fitness values; None when it is numerically zero, which turns margins off."""
    fitness = np.asarray(fitness, dtype=np.float64)
    scale = float(fitness.std(ddof=0))
    if scale <= rel_tol * max(1.0, abs(float(fitness.mean()))):
        return None
    return scale


def mutation_count_ratio(mutants):
    """R = exp(H) / mean(n) and the largest n, where n is the mutations per variant and H the entropy of n."""
    counts = np.array([str(mutant).count(':') + 1 for mutant in mutants])
    _, frequencies = np.unique(counts, return_counts=True)
    probabilities = frequencies / frequencies.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()) / counts.mean()), int(counts.max())


def scheduled_rho(rho0, shot, k0, eta):
    return float(rho0 * (np.log(1 + k0) / np.log(1 + shot)) ** eta)


def _gap_margin(gaps, scale, m_max):
    if scale is None:
        return torch.zeros_like(gaps)
    normalized = gaps.abs() / scale
    return m_max * normalized / (1 + normalized)


def ancor_loss_sums(scores, golden_score, cut_off, beta, scale, m_max):
    """Sums and valid counts of the relative (pair) and directional (wild type) losses on one batch."""
    scores = scores.to(torch.float32)
    golden_score = golden_score.to(device=scores.device, dtype=torch.float32)
    first, second = torch.triu_indices(scores.shape[0], scores.shape[0], offset=1, device=scores.device)

    pair_gaps = golden_score[first] - golden_score[second]
    wt_gaps = golden_score - cut_off
    pair_valid = pair_gaps != 0
    wt_valid = wt_gaps != 0
    pair_separation = beta * torch.sign(pair_gaps) * (scores[first] - scores[second])
    wt_separation = beta * torch.sign(wt_gaps) * scores

    rel_sum = (F.softplus(_gap_margin(pair_gaps, scale, m_max) - pair_separation) * pair_valid).sum()
    # a variant counts once per pair it appears in
    endpoints = torch.cat([first, second])
    endpoint_valid = wt_valid[endpoints]
    dir_sum = (F.softplus(_gap_margin(wt_gaps, scale, m_max) - wt_separation)[endpoints] * endpoint_valid).sum()
    return rel_sum, dir_sum, float(pair_valid.sum().item()), float(endpoint_valid.sum().item())


def direction_weight(norm_rel_sq, norm_dir_sq, dot_rel_dir, rho, alpha_min, alpha_max, eps=1e-8, dir_tol=1e-12):
    """Weight of the directional loss for one step: the projection coefficient plus the clipped adaptive coefficient."""
    p = -dot_rel_dir / norm_dir_sq if dot_rel_dir < 0 and norm_dir_sq > dir_tol else 0.0
    norm_rel_proj = max(norm_rel_sq + 2 * p * dot_rel_dir + p ** 2 * norm_dir_sq, 0.0) ** 0.5
    alpha = rho * norm_rel_proj / (max(norm_dir_sq, 0.0) ** 0.5 + eps)
    return p + min(max(alpha, alpha_min), alpha_max)


def _observed_token_log_probs(logits, seq):
    token_ids = seq.to(device=logits.device, dtype=torch.long).unsqueeze(-1)
    return torch.log_softmax(logits, dim=-1).gather(dim=-1, index=token_ids).squeeze(-1)


def kl_loss(policy_logits, reference_logits, seq, att_mask):
    """KL regularizer towards the pretrained model, on the observed tokens."""
    policy_log_probs = _observed_token_log_probs(policy_logits, seq)
    reference_log_probs = _observed_token_log_probs(reference_logits, seq)
    mask = att_mask.to(device=policy_logits.device, dtype=policy_log_probs.dtype)
    token_loss = policy_log_probs.exp() * (policy_log_probs - reference_log_probs)
    return ((token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()
