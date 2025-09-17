# HAPO Implementation

## Configuration

### Adaptive Temperature Sampling

An example configuration in config.yaml

```yaml
actor_rollout_ref::
  rollout:
    adaptive_temperature: True
    temperature_tau: 0.05
```

`adaptive_temperature` and `temperature_tau` specify whether to enable adaptive_temperature and the range for dynamic temperature adjustment.

Core relevant code:

```python
def compute_entropy_based_temperature(
    logits: torch.Tensor,
    input_temperature: torch.Tensor,
    adaptive_temperature: bool = False,
    temperature_tau: float = 0.05,
    norm_entropys: tuple = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute dynamic temperature from entropy of logits."""
    if adaptive_temperature and norm_entropys is not None:
        median, std, scale_pos, scale_neg = norm_entropys
      
        logits = logits.to(torch.float)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.log(entropy.clamp_min(eps))
                  
        h = (entropy - median) / (std + eps)
        h_hat = torch.where(h > 0, h * scale_pos, torch.where(h <= 0, h * scale_neg, torch.zeros_like(h))).clamp(-1, 1)
                  
        dynamic_temperature = input_temperature * (1.0 + h_hat * temperature_tau)
                  
        return dynamic_temperature

    return input_temperature
```

### Token Level Group Average

An example configuration in your sh script

```bash
adv_estimator=hapo
```

When `adv_estimator` is set to `hapo`, advantage calculation will be performed at the token-level within each batch.

Core relevant code:

```python
@register_adv_est(AdvantageEstimator.HAPO)
def compute_hapo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    entropys: torch.Tensor | None = None,
):
    scores_grpo, _ = compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo)
    
    scores = token_level_rewards.sum(dim=-1)
    scores = scores.unsqueeze(-1) * response_mask
    
    id2score = defaultdict(list)
    id2entropy = defaultdict(list)
    id2mask = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2entropy[index[i]].append(entropys[i])
            id2mask[index[i]].append(response_mask[i])
        for idx in id2score:
            id2score[idx], masks = torch.stack(id2score[idx]), torch.stack(id2mask[idx])

            valid_vals = id2score[idx].masked_select(masks.bool())
            id2mean[idx] = valid_vals.mean()
            id2std[idx] = valid_vals.std(unbiased=False)
        
        for idx in id2score:
            group_idx = [i for i, x in enumerate(index) if x == idx]
            grp_scores = scores[group_idx]
            
            if norm_adv_by_std_in_grpo:
                normed = (grp_scores - id2mean[idx]) / (id2std[idx] + epsilon)
            else:
                normed = grp_scores - id2mean[idx]

            # Here, we apply smooth score. If the normed advantage is too large, we use the GRPO advantage instead.
            if torch.any(torch.abs(normed) >= 5):
                scores[group_idx] = scores_grpo[group_idx]
            else:
                scores[group_idx] = normed

    return scores, scores
```

### Differential Advantage Redistribution

An example configuration in config.yaml

```yaml
actor_rollout_ref:
  actor:
    adaptive_advantage: True 
    adv_beta: 1.0
```

`adaptive_advantage` and `adv_beta` specify whether to enable advantage redistribution and the strength of adaptive advantage differential. Since we directly use entropy for regulation and to avoid introducing hyperparameters, we set `adv_beta` to 1.

Core relevant code:

```python
def ada_advantage(advantages: torch.Tensor, h_hat: torch.Tensor, ratio: torch.Tensor, cliprange_low: torch.Tensor, cliprange_high: torch.Tensor, adv_beta: float = 1.0): 
    advantages_ent = advantages * (1 + adv_beta * h_hat)
    
    neutral_left, neutral_right = 1.0 - cliprange_low / 2, 1.0 + cliprange_high / 2
    pos_adv = torch.where((ratio > neutral_left) & (ratio < neutral_right), advantages, advantages_ent)
    neg_adv = torch.where((ratio > neutral_left) & (ratio < neutral_right), advantages_ent, advantages)
    
    advantages = torch.where(h_hat > 0, pos_adv, neg_adv)
        
    return advantages
```

### Asymmetric Adaptive Clipping

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    adaptive_clip: True 
    clip_alpha: 1.0
```

`adaptive_clip` and `adaptive_clip` specify whether to enable adaptive clipping and the strength of adaptive clipping boundaries. Since we directly use entropy for regulation and to avoid introducing hyperparameters, we set `clip_alpha` to 1.

Core relevant code:

```python
def ada_clip(cliprange_low: float = 0.2, cliprange_high: float = 0.2, h_hat: torch.Tensor = None, clip_alpha: float = 1.0):
    cliprange_low  = torch.full_like(h_hat, cliprange_low)
    cliprange_high = torch.full_like(h_hat, cliprange_high)
        
    cliprange_low_ent = cliprange_low * (1 - h_hat * clip_alpha)
    cliprange_high_ent = cliprange_high * (1 + h_hat * clip_alpha)
    
    low_entropy_mask = h_hat <= 0
    high_entropy_mask = h_hat > 0
        
    cliprange_low[low_entropy_mask] = cliprange_low_ent[low_entropy_mask]
    cliprange_high[high_entropy_mask] = cliprange_high_ent[high_entropy_mask]
        
    return cliprange_low, cliprange_high
```

## Training Notes
1. We found that the HAPO method is highly sensitive to overlong buffer penalty. Since the overlong buffer penalty penalizes length, this may cause the model to be confused about why it doesn't receive positive rewards even when answering correctly. If you observe a sharp decline in response length during training, you can disable the overlong buffer penalty.
2. If you find model training collapse, consider reducing temperature_tau. We found that Qwen series models are very sensitive to temperature. Lowering temperature_tau helps improve stability. 