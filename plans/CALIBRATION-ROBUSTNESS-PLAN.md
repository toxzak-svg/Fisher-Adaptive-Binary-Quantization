# Calibration Robustness & Downstream Flexibility Plan

**Date:** 2026-05-16
**Status:** In Progress
**Concerns Address:** Fisher calibration narrowness, context shape mismatch, downstream flexibility with adapters

---

## 0. Executive Summary

FABQ-RC's core vulnerability is that **Fisher importance is only as good as the calibration distribution**. If calibration data is narrow, "low Fisher" may mean "not activated by this eval format" rather than "safe to binarize."

---

## 1. Multi-Domain Fisher Scoring (Never Implemented)

### 1.2 Stratified Calibration Set

Replace the single C4 slice with a **stratified multi-domain calibration set**:

| Domain | Source | Samples | Seq Len | Rationale |
|--------|--------|---------|---------|-----------|
| Natural language | C4 / WikiText-2 | 512 | 128-256 | General LM behavior |
| Code | The Stack / CodeAlpaca | 256 | 256-512 | Activates coding-specific channels |
| Math | GSM8K / MathQA | 256 | 128-256 | Activates reasoning chains |
| Long context | PG-19 / gov-report | 128 | 1024+ | Activates long-range attention |
| Instruction-following | Anthropic HH / Dolly | 256 | 128-256 | Activates formatting/special tokens |

Total: ~1408 samples (manageable for one forward+backward pass).

### 1.3 Per-Domain Fisher + Aggregation

```python
def compute_multi_domain_fisher(model, domain_loaders, device):
    """
    Compute Fisher scores per domain, then aggregate.
    
    Returns:
        per_domain: dict[domain_name, dict[layer_name, Tensor]]
        aggregated: dict[layer_name, Tensor]  (after aggregation)
        domain_spikes: dict[layer_name, Tensor]  (channels that spike in any domain)
    """
    per_domain = {}
    for domain_name, loader in domain_loaders.items():
        fisher_acc = FisherAccumulator(model)
        per_domain[domain_name] = fisher_acc(loader, device)
    
    # Aggregation strategies — pick one or ensemble:
    aggregated = {}
    domain_spikes = {}
    for layer_name in per_domain[next(iter(per_domain))]:
        scores = torch.stack([per_domain[d][layer_name] for d in per_domain])
        
        # Strategy A (default): Max-aggregation — protect channels important in ANY domain
        aggregated[layer_name] = scores.max(dim=0).values
        
        # Strategy B: Top-2 average — consensus across multiple domains
        # sort scores descending, take average of top 2 per channel
        top2 = scores.topk(2, dim=0).values.mean(dim=0)
        
        # Domain spikes: channels where Fisher is >> domain mean
        domain_spikes[layer_name] = (
            scores.max(dim=0).values / scores.mean(dim=0)
        )
    
    return per_domain, aggregated, domain_spikes
```

### 1.4 Domain-Spike Protection

Channels that **spike in any single domain** (even if low-averaged) get automatic protection. This is the direct fix for "safe to crush" channels that carry rare but critical behavior.

```python
def allocate_with_spike_protection(
    aggregated_fisher, domain_spikes, int4_fraction=0.05,
    spike_threshold=2.0  # 2x domain-mean Fisher = protected
):
    """
    Standard FABQ-RC allocation, but channels that spike in any domain 
    get promoted to int4 regardless of their aggregated rank.
    
    spike_threshold: channels with domain_spike ratio > this get promoted.
    """
    allocation = {}
    for name, fisher in aggregated_fisher.items():
        out_c = fisher.shape[0]
        n_int4 = max(1, int(out_c * int4_fraction))
        
        # Base allocation by aggregated Fisher
        order = torch.argsort(fisher, descending=True)
        base_alloc = {}
        for rank, ch in enumerate(order):
            base_alloc[int(ch)] = 'int4' if rank < n_int4 else 'binary'
        
        # Spike protection: promote high-spike channels to int4
        spikes = domain_spikes[name]
        spike_channels = (spikes > spike_threshold).nonzero().squeeze(-1)
        for ch in spike_channels.tolist():
            base_alloc[ch] = 'int4'
        
        allocation[name] = base_alloc
    return allocation
```

### 1.5 Quick Validation

Check Fisher rank **agreement across domains** using Spearman correlation:

```python
def validate_fisher_consistency(per_domain):
    """Spearman rank correlation between domain Fisher pairs."""
    domains = list(per_domain.keys())
    for layer_name in per_domain[domains[0]]:
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                rho = spearmanr(
                    per_domain[d1][layer_name].numpy(),
                    per_domain[d2][layer_name].numpy()
                ).correlation
                if rho < 0.5:
                    print(f"WARNING: {layer_name}: {d1} vs {d2} rho={rho:.3f}")
```

Low cross-domain rank agreement → strong signal that single-domain calibration is unreliable.

---

## 2. Context Shape Sensitivity Analysis

### 2.1 Problem

Current calibration uses seq_len=128 with padding to max_length. The base model was trained on rolling windows with diverse context shapes. Fisher scores computed on short, clean sequences may miss channels that activate in:
- Middle-of-sequence (where ambiguity is highest)
- Long-range dependencies (beyond 128 tokens)
- Prefix positions (where model is most uncertain)

### 2.2 Context Variation Experiments

Systematically vary context characteristics to measure Fisher stability:

```python
context_configs = [
    # (seq_len, position_in_context, label)
    (64,  'prefix',  "short prefix — clean start"),
    (128, 'prefix',  "standard prefix — current default"),
    (512, 'prefix',  "medium prefix — typical eval length"),
    (128, 'middle',  "short middle — extract from longer sequence"),
    (512, 'middle',  "medium middle — ambiguous context"),
    (2048, 'middle', "long middle — full dependency range"),
]
```

### 2.3 "Format-Sensitive" Channel Detection

Channels whose Fisher rank shifts significantly across context configurations. These are the channels most likely to be falsely classified as "low Fisher" by a narrow calibration.

```python
def detect_format_sensitive_channels(model, configs, device):
    """
    For each context config, compute Fisher and identify channels 
    whose rank changes the most.
    
    Returns: dict[layer_name, Tensor] of rank volatility score
    """
    config_fishers = {}
    for seq_len, position, label in configs:
        loader = build_context_loader(seq_len, position, n_samples=128)
        fisher_acc = FisherAccumulator(model)
        config_fishers[label] = fisher_acc(loader, device)
    
    # Per-channel rank volatility = std of rank across configs
    volatility = {}
    for layer_name in config_fishers[configs[0][2]]:
        ranks = []
        for label, fisher in config_fishers.items():
            rank = fisher[layer_name].argsort(descending=True).float()
            ranks.append(rank)
        rank_stack = torch.stack(ranks, dim=0)
        volatility[layer_name] = rank_stack.std(dim=0)
    
    return volatility
```

### 2.4 Context Protection

Channels with high rank volatility get protected (promoted to int4), because their "low Fisher" status is an artifact of calibration distribution rather than genuine low importance.

---

## 3. Downstream Flexibility Analysis

### 3.1 Problem

Binary quantization removes all magnitude information from 95% of channels. This raises concrete downstream concerns:

1. **LoRA adaptation**: Do binary weights require larger-rank adapters to compensate for lost expressivity?
2. **Featurewise magnitude**: Does reasoning capability degrade because magnitude ratios between channels are lost?
3. **Hook-based interventions**: Can activation steering / representation engineering work on binarized models?

### 3.2 LoRA Recovery Experiment

```python
def loora_recovery_experiment(base_model, fisher_scored_model, calibration_data):
    """
    Compare how much LoRA rank is needed to recover perplexity from 
    FABQ-RC quantization vs uniform quantization baselines.
    
    Hypothesis: FABQ-RC binary weights require 2x LoRA rank vs int4
    at the same bpw to match perplexity. This quantifies the "adapter cost"
    of binary freezing.
    """
    results = []
    
    for rank in [4, 8, 16, 32, 64]:
        for quant_method in ['fabq_binary', 'uniform_int4', 'uniform_int2']:
            model = load_quantized(quant_method)
            
            # Train LoRA adapter on held-out data
            lora_model = inject_lora(model, rank=rank)
            ppl = train_and_evaluate(lora_model, calibration_data)
            
            results.append({
                'quant_method': quant_method,
                'lora_rank': rank,
                'ppl': ppl,
                'total_params': count_quant_params(quant_method) + count_lora_params(rank)
            })
    
    return results
```

### 3.3 Magnitude Information Loss Measurement

Quantify what information is lost when weights are binarized:

```python
def magnitude_loss_analysis(original_weights, binary_weights):
    """
    Measure what the binary representation discards.
    
    Metrics:
    1. Sign agreement: what fraction of sign bits match?
    2. Magnitude rank: how well does binary ±scale preserve channel ordering?
    3. Ratio distortion: how much do inter-channel magnitude ratios change?
    """
    sign_agreement = (original_weights > 0) == (binary_weights > 0)
    
    # Magnitude rank correlation
    orig_mag = original_weights.abs().sum(dim=-1)  # per-channel total magnitude
    binary_mag = binary_weights.abs().sum(dim=-1)
    magnitude_rank_rho = spearmanr(orig_mag, binary_mag)
    
    # Ratio distortion — key for attention logit computation
    orig_ratios = original_weights / (original_weights.norm(dim=-1, keepdim=True) + 1e-8)
    binary_ratios = binary_weights / (binary_weights.norm(dim=-1, keepdim=True) + 1e-8)
    ratio_distortion = (orig_ratios - binary_ratios).norm(dim=-1).mean()
    
    return {
        'sign_agreement': sign_agreement.float().mean().item(),
        'magnitude_rank_rho': magnitude_rank_rho,
        'ratio_distortion': ratio_distortion.item()
    }
```

### 3.4 Adaptation-Friendly Allocation

For deployment scenarios where adapters are expected, add a **"future adaptation budget"** — reserve a small fraction of channels (e.g., 1-2%) as int4 specifically near the Fisher boundary, so they retain magnitude information for later fine-tuning:

```python
def allocate_with_adaptation_margin(fisher, int4_fraction=0.05, margin=0.01):
    """
    Standard allocation, but with a 'margin' of extra channels kept at int4
    around the binary/int4 boundary. These channels have ambiguous Fisher
    importance — they might be needed for downstream adaptation.
    """
    n_int4 = max(1, int(len(fisher) * (int4_fraction + margin)))
    n_core = max(1, int(len(fisher) * int4_fraction))
    
    order = torch.argsort(fisher, descending=True)
    
    allocation = {}
    for rank, ch in enumerate(order):
        if rank < n_core:
            allocation[int(ch)] = 'int4'  # definitely important
        elif rank < n_int4:
            allocation[int(ch)] = 'int4'  # adaptation margin
        else:
            allocation[int(ch)] = 'binary'
    
    return allocation
```

### 3.5 Training-Free Reasoning

Test whether binary quantization preserves enough structure for training-free adaptation methods:

- **Activation steering**: Can binary weights still support activation addition for behavior control?
- **Representation reading**: Are hidden state magnitudes still meaningful for probing?
- **Logit lens**: Do early-exit logits still work?

These are lightweight sanity checks that don't require training.

---

## 4. Methodological Safeguards

### 4.1 Fisher Confidence Intervals

Bootstrap Fisher scores over calibration subsets to quantify uncertainty:

```python
def fisher_confidence(model, loader, n_bootstrap=50, sample_frac=0.8):
    """
    Bootstrap Fisher scores to estimate confidence intervals per channel.
    Channels with wide CI are unreliable — their precision assignment 
    depends on calibration luck.
    """
    all_scores = []
    for _ in range(n_bootstrap):
        subset = torch.utils.data.Subset(loader, 
            np.random.choice(len(loader), int(len(loader)*sample_frac)))
        fisher_acc = FisherAccumulator(model)
        scores = fisher_acc(DataLoader(subset), device)
        all_scores.append(scores)
    
    # Per-channel CI width
    stability = {}
    for layer_name in all_scores[0]:
        stacked = torch.stack([s[layer_name] for s in all_scores], dim=0)
        ci_width = stacked.std(dim=0) / stacked.mean(dim=0)  # CV
        stability[layer_name] = ci_width
    
    return stability
```

### 4.2 Fisher Stability Index

A single scalar per model quantifying how much the Fisher ranking depends on calibration choices:

```python
def fisher_stability_index(per_domain_fisher):
    """
    FSI = fraction of channels that stay in the top 5% across ALL domains.
    High FSI (>0.8) → Fisher is robust. Low FSI (<0.3) → highly calibration-dependent.
    """
    domains = list(per_domain_fisher.keys())
    top_sets = []
    for domain in domains:
        for layer_name, fisher in per_domain_fisher[domain].items():
            n_int4 = max(1, int(len(fisher) * 0.05))
            top_set = set(fisher.argsort(descending=True)[:n_int4].tolist())
            top_sets.append((domain, layer_name, top_set))
    
    # Aggregate: what fraction of top-5% channels are common across all domains?
    layer_intersection = {}
    for layer_name in [l for d, l, s in top_sets]:
        sets = [s for d, l, s in top_sets if l == layer_name]
        intersection = set.intersection(*sets) if sets else set()
        n = len(next(iter(sets))) if sets else 1
        layer_intersection[layer_name] = len(intersection) / max(n, 1)
    
    return np.mean(list(layer_intersection.values()))
```

### 4.3 Precision Allocation Validation Protocol

Before finalizing any model's precision allocation:

1. **Check cross-domain Fisher agreement** (Spearman rho > 0.5 across domain pairs)
2. **Check context stability** (rank volatility < threshold)
3. **Check bootstrap CI width** (CV < 0.3 per channel)
4. **Run Fisher Stability Index** (FSI > 0.5)
5. **Audit "boundary channels"** — inspect channels near the int4/binary cutoff for domain spikes

### 4.4 Boundary Channel Audit

Channels at the precision boundary (rank ~5-10% by Fisher) are the most risky — they get different treatment depending on small calibration changes. Log them for human review:

```python
def audit_boundary_channels(fisher, domain_spikes, n_boundary=0.05):
    """
    For channels near the int4/binary cutoff:
    - Show their Fisher rank in each domain
    - Show their domain spike ratio
    - Recommend promotion if any domain would rank them top-5%
    """
    n_check = max(1, int(len(fisher) * n_boundary))
    order = torch.argsort(fisher, descending=True)
    boundary = order[n_check:2*n_check]  # channels just below cutoff
    
    for ch in boundary:
        spike = domain_spikes[ch].item()
        print(f"  ch {ch:5d} | aggregated rank: {order.tolist().index(ch):4d} "
              f"| domain spike: {spike:.2f}x")
```

---

## 5. Implementation Plan

### Phase 1: Multi-Domain Fisher Infrastructure (Priority: High)

- [ ] Build `MultiDomainFisherAccumulator` class with per-domain accumulation
- [ ] Implement aggregation strategies (max, top-k, weighted)
- [ ] Build domain-spike protection allocation (`allocate_with_spike_protection`)
- [ ] Add calibration dataset loading for 4+ domains
- [ ] **Test**: Compare per-domain Fisher distributions for TinyLlama

### Phase 2: Context Shape Profiling (Priority: High)

- [ ] Implement context variation loader (seq_len x position matrix)
- [ ] Implement `detect_format_sensitive_channels()`
- [ ] Measure Fisher rank volatility across 3+ context configurations
- [ ] **Test**: Does rank volatility correlate with downstream performance?

### Phase 3: Fisher Robustness Diagnostics (Priority: Medium)

- [ ] Implement bootstrap CI for Fisher scores
- [ ] Implement Fisher Stability Index
- [ ] Implement boundary channel audit
- [ ] Add validation protocol as notebook cell block
- [ ] **Test**: Run diagnostics on existing FABQ-RC calibration pipeline

### Phase 4: Downstream Flexibility (Priority: Medium-High)

- [ ] Implement LoRA recovery experiment
- [ ] Implement magnitude information loss measurement
- [ ] Implement adaptation-margin allocation
- [ ] **Test**: Compare LoRA recovery for FABQ-RC vs int4 baselines
- [ ] **Test**: Compare downstream task accuracy with/without adaptation margin

### Phase 5: Training-Free Reasoning Diagnostics (Priority: Low)

- [ ] Activation steering test on binary-quantized model
- [ ] Representation reading / probing test
- [ ] Logit lens sanity check

---

## 6. Success Criteria

| Criterion | Target | Measured By |
|-----------|--------|-------------|
| Cross-domain Fisher agreement | Spearman rho > 0.7 across all domain pairs | `validate_fisher_consistency()` |
| Fisher Stability Index | FSI > 0.5 | `fisher_stability_index()` |
| Context rank volatility | std(rank) < 10% of total channels | `detect_format_sensitive_channels()` |
| Bootstrapped CI width | CV < 0.3 per channel | `fisher_confidence()` |
| Domain-spike protection | Catches >90% of truly important low-Fisher channels | Synthetic benchmark |
| LoRA recovery gap | FABQ-RC requires <= 2x LoRA rank vs int4 at same bpw | `lora_recovery_experiment()` |

---

## 7. Relationship to Existing Pipeline

```
Current:
  C4 (256 samples, seq_len=128) → Fisher → Allocation

Proposed:
  ┌─────────────────────────────────────────────────────────┐
  │  Multi-Domain Loader                                    │
  │  ├── C4 (512, seq=256)     ──┐                          │
  │  ├── Code (256, seq=512)   ──┤                          │
  │  ├── Math (256, seq=256)   ──┤── Per-Domain Fisher      │
  │  ├── Long (128, seq=1024)  ──┤                          │
  │  └── Instruct (256, seq=256) ─┘                          │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Aggregation + Diagnostics                              │
  │  ├── Max-aggregation (any-domain protection)            │
  │  ├── Domain-spike detection                             │
  │  ├── Bootstrap CI / FSI                                 │
  │  └── Boundary audit                                     │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Robust Allocation                                      │
  │  ├── Standard int4/binary split                         │
  │  ├── + Spike-promoted channels                          │
  │  ├── + Adaptation margin channels                       │
  │  └── + Audit-verified boundary                          │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Downstream Validation                                  │
  │  ├── LoRA recovery test (train-free sanity)             │
  │  ├── Magnitude info loss measurement                    │
  │  └── Training-free reasoning check                      │
  └─────────────────────────────────────────────────────────┘
```

No changes needed to Stages 3-4 (adaptive blocksize, residual codebook). They operate on the allocation produced by Stage 2, which is exactly what we're hardening.

---

## 8. Open Questions

1. **Domain count vs compute cost** — 5 domains × 1408 samples is ~5.5x the current calibration cost. Is that acceptable, or should we use a smaller subset? (Estimate: 1408 forward+backward passes on TinyLlama ≈ 2-3 minutes on a T4.)

2. **Aggregation strategy** — Max-aggregation is conservative (more channels protected). Should we make it configurable, or have domain-weighted averaging for production use?

3. **Adaptation margin size** — 1% is a guess. Should it be determined by the Fisher rank gradient at the boundary (high gradient = more margin needed)?

4. **Training-free reasoning dependence** — If activation steering fails on binary models, does that mean
"""
