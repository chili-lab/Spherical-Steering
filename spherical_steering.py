"""
Spherical Steering: Core Algorithm

This module implements the Spherical Steering intervention method for 
steering language model hidden states toward truthful directions using
von Mises-Fisher (vMF) distributions on the unit sphere.

Key Components:
- spherical_geometric_logic: Core steering logic for single vectors
- baukit_hook_fn: Hook function compatible with baukit TraceDict
- get_spherical_intervention: Factory function for creating steering hooks
"""

import torch
import torch.nn.functional as F
from functools import partial


def spherical_geometric_logic(x, mu_T, mu_H, kappa, alpha, beta):
    """
    Core spherical steering logic for a single hidden state vector.
    
    This function implements the geometric steering operation:
    1. Compute vMF probabilities for truthful (T) and hallucination (H) prototypes
    2. If hallucination probability exceeds threshold, apply steering
    3. Steering rotates the vector toward the truthful prototype on the sphere
    
    Args:
        x: Hidden state vector [D]
        mu_T: Truthful prototype (unit vector) [D]
        mu_H: Hallucination prototype (unit vector) [D]
        kappa: vMF concentration parameter (higher = sharper decisions)
        alpha: Maximum steering strength (0 to 1)
        beta: Threshold for triggering steering (p_H - p_T > beta)
    
    Returns:
        x_new: Steered hidden state vector [D]
        triggered: Boolean indicating if steering was applied
    """
    orig_dtype = x.dtype
    x = x.float()
    mu_T = mu_T.float()
    mu_H = mu_H.float()
    
    # Preserve original norm for rescaling
    orig_norm = x.norm(p=2).clamp_min(1e-12)
    x_hat = x / orig_norm 
    
    # Compute vMF log-likelihoods (proportional to cosine similarity)
    cos_T = torch.dot(x_hat, mu_T).clamp(-1, 1)
    cos_H = torch.dot(x_hat, mu_H).clamp(-1, 1)
    
    # Softmax to get probabilities
    logits = torch.stack([kappa * cos_T, kappa * cos_H])
    probs = F.softmax(logits, dim=0)
    p_T, p_H = probs[0], probs[1]
    
    # Check steering condition
    delta = p_H - p_T
    
    if delta <= beta:
        # No steering needed
        return x.to(orig_dtype), False
    
    # Compute steering strength (linear interpolation above threshold)
    t = alpha * (delta - beta) / (1.0 - beta)
    t = torch.clamp(t, 0.0, 1.0)
    
    # Compute angle from truthful prototype
    theta = torch.acos(cos_T)
    if theta < 1e-4:
        # Already very close to mu_T
        return x.to(orig_dtype), False
    
    # Compute new angle (rotate toward mu_T)
    theta_new = (1.0 - t) * theta
    
    # Spherical interpolation (SLERP-like)
    sin_theta = torch.sin(theta)
    u = (x_hat - cos_T * mu_T) / sin_theta  # Orthogonal component
    
    x_new_hat = torch.cos(theta_new) * mu_T + torch.sin(theta_new) * u
    x_new = x_new_hat * orig_norm  # Restore original norm
    
    return x_new.to(orig_dtype), True


def baukit_hook_fn(output, layer_name, mu_T, mu_H, kappa, alpha, beta, stats=None, start_idx=None):
    """
    Hook function for use with baukit TraceDict.
    
    This function modifies hidden states in-place during forward pass.
    It supports range-based intervention for scoring tasks.
    
    Args:
        output: Layer output (tuple or tensor) from transformer layer
        layer_name: Name of the layer being hooked (unused, required by baukit)
        mu_T: Truthful prototype [D]
        mu_H: Hallucination prototype [D]
        kappa: vMF concentration parameter
        alpha: Maximum steering strength
        beta: Steering threshold
        stats: Optional dict to track steering statistics
        start_idx: Starting position for intervention (None = last token only)
    
    Returns:
        Modified output with steered hidden states
    """
    if isinstance(output, tuple):
        h_hidden = output[0]  # [Batch, Seq, Dim]
    else:
        h_hidden = output
        
    device = h_hidden.device
    
    # Ensure prototypes are on the correct device
    if not isinstance(mu_T, torch.Tensor):
        mu_T = torch.tensor(mu_T, device=device)
    else:
        mu_T = mu_T.to(device)
    if not isinstance(mu_H, torch.Tensor):
        mu_H = torch.tensor(mu_H, device=device)
    else:
        mu_H = mu_H.to(device)

    batch_size, seq_len, _ = h_hidden.shape
    
    # Determine intervention range
    if start_idx is None:
        # Default: only last token (for generation)
        range_to_steer = [seq_len - 1]
    else:
        # Range-based intervention (for MC scoring)
        safe_start = max(0, min(start_idx, seq_len - 1))
        range_to_steer = range(safe_start, seq_len)

    # Apply steering to each position in range
    for i in range(batch_size):
        for t in range_to_steer:
            vec = h_hidden[i, t, :].clone()
            
            modified_vec, triggered = spherical_geometric_logic(
                vec, mu_T, mu_H, kappa, alpha, beta
            )
            
            h_hidden[i, t, :] = modified_vec
            
            if stats is not None:
                stats['total'] += 1
                if triggered:
                    stats['steered'] += 1

    if isinstance(output, tuple):
        return (h_hidden,) + output[1:]
    else:
        return h_hidden


def get_spherical_intervention(mu_T, mu_H, kappa=20.0, alpha=0.15, beta=0.1, stats=None):
    """
    Factory function to create a spherical steering hook.
    
    Returns a partially-applied hook function that can be used with baukit.
    The returned function accepts an additional `start_idx` parameter for
    range-based intervention.
    
    Args:
        mu_T: Truthful prototype [D]
        mu_H: Hallucination prototype [D]
        kappa: vMF concentration parameter (default: 20.0)
        alpha: Maximum steering strength (default: 0.15)
        beta: Steering threshold (default: 0.1)
        stats: Optional dict to track steering statistics
    
    Returns:
        Hook function compatible with baukit TraceDict
    
    Example:
        >>> hook_fn = get_spherical_intervention(mu_T, mu_H, kappa=20, alpha=0.6, beta=-0.05)
        >>> with TraceDict(model, [layer_name], edit_output=partial(hook_fn, start_idx=0)):
        ...     outputs = model(input_ids)
    """
    return partial(
        baukit_hook_fn, 
        mu_T=mu_T, 
        mu_H=mu_H, 
        kappa=kappa, 
        alpha=alpha, 
        beta=beta,
        stats=stats
    )
