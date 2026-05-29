# EBQ: Error-Budget Quantization

**Date:** 2026-05-04
**Status:** Planned
**Goal:** Global perplexity-constrained bit allocation for RAM-only LLM inference

---

## Problem Statement

All existing quantization methods use **local** decisions. EBQ treats the entire model as a single optimization problem under a perplexity budget.

---

## 1. Problem Statement

All existing quantization methods use **local** decisions:
- GPTQ: per-channel or per-group, same bit-width for all layers
- AWQ: selects "salient" weights (1%) for protection, same bits for rest
- QLoRA: uniform 4-bit, different only by layer type (attention vs MLP)

**The problem:** Local decisions don't optimize for global perplexity. A 4-bit quantization in layer 12 might matter more than the same 4-bit decision in layer 24.

**The insight:** Treat the entire model as a single optimization problem under a perplexity budget.

---

## 2. Core Algorithm: Global Bit Allocation

### 2.1 Problem Formulation

For a model with M modules and K possible quantization configurations per module:

```
Decision variables:
  x[m,c] ∈ {0,1}  for module m, config c (one config per module)

Constraints:
  Σ[m] size[m, config[m]] ≤ S_target           (size budget)
  PPL(fp16) × (1 + ε) ≥ PPL(x)               (perplexity budget)
  
Objective: none (feasibility problem)
```

Each `config` specifies:
- Bit-width: 2, 3, 4, 6, 8 bits (or FP16 for critical)
- Quantizer type: GPTQ, per-tensor, per-channel, per-group
- Group size: 32, 64, 128 (for grouped schemes)
- Salient protection: which channels stay FP16 (AWQ-style)

### 2.2 Sensitivity Computation

```python
def compute_module_sensitivity(model, module_name, calibration_loader, device):
    """
    Measure how much perplexity degrades when this module is quantized.
    
    Returns: dict of {config: (ppl_delta, size_delta)}
    """
    module = get_module(model, module_name)
    fp16_outputs = run_forward_pass(model, calibration_loader, device)
    
    sensitivities = {}
    
    for bits in [2, 3, 4, 6, 8]:
        for quantizer_type in ['per_channel_gptq', 'per_tensor', 'per_group_64']:
            # Quantize only this module
            quantized_module = quantize_module(module, bits, quantizer_type)
            
            # Run forward pass and measure ppl
            quantized_outputs = run_forward_pass(model, calibration_loader, device)
            ppl_delta = compute_ppl_delta(fp16_outputs, quantized_outputs)
            size_delta = compute_size_delta(module, bits, quantizer_type)
            
            sensitivities[f"{bits}b_{quantizer_type}"] = (ppl_delta, size_delta)
    
    return sensitivities
```

### 2.3 Greedy Knapsack Allocation

```python
def error_budget_allocation(model, module_sensitivities, 
                            S_target, epsilon_ppl, base_ppl):
    """
    Greedy allocation under size and perplexity budgets.
    
    Args:
        model: the model
        module_sensitivities: {module_name: {config: (ppl_delta, size_delta)}}
        S_target: target size in bytes
        epsilon_ppl: allowed PPL increase (e.g., 0.05 for 5%)
        base_ppl: FP16 baseline perplexity
    
    Returns: {module_name: best_config} assignments
    """
    PPL_limit = base_ppl * (1 + epsilon_ppl)
    
    # Start with safe (high-bit) config for all modules
    allocation = {}
    for module_name in module_sensitivities:
        # Find the highest-bit (safest) config as starting point
        safest_config = max(
            module_sensitivities[module_name].keys(),
            key=lambda c: int(c.split('b')[0])  # highest bits
        )
        allocation[module_name] = safest_config
    
    current_size = sum_module_sizes(allocation, module_sensitivities)
    current_ppl_estimate = estimate_ppl(allocation, module_sensitivities)
    
    # Priority queue: configs sorted by "bits saved per ppl increase"
    pq = PriorityQueue()
    
    for module_name in module_sensitivities:
        current_config = allocation[module_name]
        current_bits = int(current_config.split('b')[0])
        
        for candidate_config in module_sensitivities[module_name]:
            candidate_bits = int(candidate_config.split('b')[0])
            
            if candidate_bits >= current_bits:
                continue  # only consider downgrades
            
            delta_ppl = module_sensitivities[module_name][candidate_config][0]
            delta_size = module_sensitivities[module_name][candidate_config][1]
            
            # Efficiency: bits saved per unit ppl increase
            if delta_ppl > 0:
                efficiency = -delta_size / delta_ppl  # negative because we save size
            else:
                efficiency = float('inf')  # no ppl cost = always good
            
            pq.push(PQItem(
                module=module_name,
                from_config=current_config,
                to_config=candidate_config,
                efficiency=efficiency,
                delta_ppl=delta_ppl,
                delta_size=delta_size
            ))
    
    # Greedy selection
    while not pq.empty():
        item = pq.pop()
        
        # Check if this upgrade stays within budgets
        new_size = current_size + item.delta_size
        new_ppl_estimate = current_ppl_estimate + item.delta_ppl
        
        if new_size > S_target:
            continue
        if new_ppl_estimate > PPL_limit:
            continue
        
        # Apply the upgrade
        allocation[item.module] = item.to_config
        current_size = new_size
        current_ppl_estimate = new_ppl_estimate
    
    return allocation
```

---

## 3. Component-Wise Sensitivity Analysis

### 3.1 What to Profile

For each linear layer `module`, we profile these components separately:

| Component | Options | Notes |
|-----------|---------|-------|
| Q_proj | 2/3/4/6/8 bits, per-channel/group | Attention query |
| K_proj | 2/3/4/6/8 bits, per-channel/group | Attention key (smaller) |
| V_proj | 2/3/4/6/8 bits, per-channel/group | Attention value |
| O_proj | 2/3/4/6/8 bits, per-channel/group | Attention output |
| Gate_proj | 2/3/4/6/8 bits, per-channel/group | MLP gate |
| Up_proj | 2/3/4/6/8 bits, per-channel/group | MLP up |
| Down_proj | 2/3/4/6/8 bits, per-channel/group | MLP down |
| LayerNorm | FP16 only | Critical for stability |
| Embeddings | 4/8 bits | Can be lower precision |

### 3.2 Layer Normalization Protection

LayerNorm and layernorm-style operations (RMSNorm, etc.) must remain FP16:
- They control dynamic range
- Instabilities in layernorm propagate to all subsequent layers
- Size savings are negligible (~0.1% of model size)

### 3.3 KV Cache Budget

At long context lengths, KV cache dominates memory. EBQ allocates KV bits separately:

```python
def profile_kv_sensitivity(model, calibration_loader, seq_lengths=[512, 2048, 8192]):
    """
    Profile how much PPL degrades at different KV cache bit-widths.
    
    Returns: dict of {seq_len: {kv_bits: ppl_delta}}
    """
    results = {}
    
    for seq_len in seq_lengths:
        fp16_kv_ppl = measure_ppl_with_kv(model, calibration_loader, seq_len, kv_bits=16)
        
        for kv_bits in [2, 3, 4, 6, 8]:
            quantized_kv_ppl = measure_ppl_with_kv(model, calibration_loader, seq_len, kv_bits=kv_bits)
            results[seq_len][kv_bits] = quantized_kv_ppl - fp16_kv_ppl
    
    return results
```

---

## 4. Storage Format for Variable-Precision Weights

### 4.1 Container Format

```python
@dataclass
class EBQContainer:
    """Variable-precision model storage format."""
    
    # Header
    magic: str = "EBQ1"  # format identifier
    version: int = 1
    n_layers: int
    hidden_size: int
    num_attention_heads: int
    
    # Per-layer metadata
    layer_configs: List[LayerConfig]
    
    # Quantized weights (packed)
    weight_data: bytes  # variable-width packed integers
    scale_data: bytes   # FP16 scales
    
    # KV cache config
    kv_bits: int
    kv_scales: bytes

@dataclass 
class LayerConfig:
    """Per-layer precision configuration."""
    layer_idx: int
    
    # Precision per component: 'q', 'k', 'v', 'o', 'gate', 'up', 'down'
    # Value is tuple of (bits, quantizer_type, group_size)
    q_config: Tuple[int, str, int]
    k_config: Tuple[int, str, int]
    v_config: Tuple[int, str, int]
    o_config: Tuple[int, str, int]
    gate_config: Tuple[int, str, int]
    up_config: Tuple[int, str, int]
    down_config: Tuple[int, str, int]
    
    # Salient channel indices (FP16 protected)
    salient_q: List[int]
    salient_k: List[int]
    salient_v: List[int]
    salient_o: List[int]
    salient_mlp: List[int]
```

### 4.2 Packing Scheme

```
For 4-bit weights with group_size=128:

Group of 128 weights → 64 bytes (packed 4-bit)
Plus 2 bytes for FP16 scale
Total: 66 bytes per 128 weights

For comparison:
FP16: 128 × 2 = 256 bytes
int8 with per-channel: 128 bytes + 128 × 2 = 384 bytes
```

---

## 5. Calibration Protocol

### 5.1 Dataset Requirements

| Use Case | Recommended Dataset | Size |
|----------|-------------------|------|
| General LLM | WikiText2 | 2M tokens |
| Code models | TheStack | 1M tokens |
| Chat models | Anthropic HH-RLHF | 500K tokens |
| Mixed | C4 | 1M tokens |

### 5.2 Calibration Procedure

```python
def calibrate_model_ebq(model, calibration_dataset, device):
    """
    Full calibration pipeline for EBQ.
    
    1. Collect per-module activations and gradients
    2. Compute sensitivity profiles
    3. Determine salient channels per module
    4. Profile KV cache sensitivity
    """
    model.eval()
    
    # Step 1: Forward passes to collect statistics
    print("Collecting activation statistics...")
    activation_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            activation_stats[name] = {
                'mean': torch.zeros_like(module.weight),
                'var': torch.zeros_like(module.weight),
                'amax': torch.zeros_like(module.weight)
            }
    
    for batch in tqdm(calibration_dataset):
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            # Track activation statistics...
    
    # Step 2: Backward pass for Fisher importance
    print("Computing Fisher importance...")
    fisher_info = compute_fisher_importance(model, calibration_dataset, device)
    
    # Step 3: Sensitivity profiling
    print("Profiling quantization sensitivity...")
    sensitivities = {}
    for name, module in tqdm(list(model.named_modules())[:10]):  # first 10 for speed
        if isinstance(module, nn.Linear):
            sensitivities[name] = compute_module_sensitivity(
                module, calibration_dataset, device
            )
    
    # Step 4: KV profiling
    print("Profiling KV cache sensitivity...")
    kv_sensitivity = profile_kv_sensitivity(model, calibration_dataset)
    
    return {
        'fisher': fisher_info,
        'sensitivities': sensitivities,
        'kv_sensitivity': kv_sensitivity
    }
```

---

## 6. EBQ vs FABQ-RC Relationship

| Aspect | EBQ | FABQ-RC |
|--------|-----|---------|
| **Approach** | Global optimization (knapsack) | Local per-layer (heuristic) |
| **Precision** | 2-8 bits continuous | 1-bit + int8 only |
| **Allocation metric** | Measured PPL delta per module | Fisher importance |
| **Codebook** | None (residual correction not included) | k-means residual codebook |
| **Blocksize** | Fixed (per quantizer type) | Adaptive per-layer |
| **Target models** | Qwen 35B, Llama 70B | TinyLlama, Mistral 7B |
| **VRAM target** | RAM-only inference | GPU inference |

### 6.1 Combined Approach: FABQ-RC + EBQ

The cleanest architecture:
1. **FABQ-RC** for binary/low-bit components (Fisher importance, adaptive blocksize, residual codebook)
2. **EBQ** for int4/int8/FP16 components (global error-budget allocation)

This gives you:
- FABQ-RC's expressive residual correction for 1-2 bit weights
- EBQ's principled global allocation for 3-8 bit weights
- Unified calibration and storage format

---

## 7. Runtime Architecture

### 7.1 CPU-Only Inference Path

```
┌──────────────────────────────────────────────────────────┐
│                      Host RAM                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │  EBQ Container (~13-16 GB for 35B model)           │  │
│  │  - Variable-precision weights                       │  │
│  │  - Scale factors                                   │  │
│  │  - Layer configs (JSON)                           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                           │
                           │ streaming dequant
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      CPU Compute                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Quantized matmul (int4/int8 SIMD)                 │  │
│  │  - AVX2/VNNI for int8                               │  │
│  │  - AVX512 + VNNI for int4                          │  │
│  │  - Thread pools for layer parallelism               │  │
│  └────────────────────────────────────────────────────┘  │
│                           │                               │
│                           │ FP16 activations             │
│                           ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Attention (FP16 softmax, FP16 KV update)          │  │
│  │  - KV cache in int4 (dequant on read)              │  │
│  │  - Attention scores in FP32 accumulator           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Optional GPU Assist

```
┌──────────────────────────────────────────────────────────┐
│                      Host RAM                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │  EBQ weights (pinned memory for fast transfer)     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                           │
                           │ PCIe (async)
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    GPU VRAM (4-8 GB)                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Layer cache: 1-2 layers resident                  │  │
│  │  - Dequantize weights to FP16                       │  │
│  │  - Compute attention + MLP                          │  │
│  │  - Quantize KV back to int4 for cache              │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 8. Experiment Plan

### Phase 1: Sensitivity Profiling (1-2 sessions)
- [ ] Load Qwen 3.5/3.6 35B in FP16
- [ ] Run calibration on WikiText2 (~10K samples)
- [ ] Profile each module's sensitivity to bit-width changes
- [ ] Profile KV cache sensitivity at different sequence lengths
- [ ] Output: sensitivity matrix (modules × configs)

### Phase 2: Allocation Experiments (2-3 sessions)
- [ ] Implement greedy knapsack allocator
- [ ] Run with 3 bpw target, 5% PPL budget
- [ ] Run with 3.5 bpw target, 3% PPL budget  
- [ ] Run with 4 bpw target, 2% PPL budget
- [ ] Output: quantization configs per layer

### Phase 3: Evaluation (1-2 sessions)
- [ ] Evaluate perplexity on WikiText2 test
- [ ] Evaluate downstream tasks (ARC, HellaSwag, etc.)
- [ ] Compare against AWQ/GPTQ baselines at same bpw
- [ ] Output: benchmark results

### Phase 4: Iterative Refinement (ongoing)
- [ ] Adjust calibration dataset composition
- [ ] Fine-tune sensitivity thresholds
- [ ] Add salient weight optimization
- [ ] Profile and optimize compute hotspots

---

## 9. Open Questions

1. **Calibration dataset size** — 10K samples enough for reliable sensitivity estimation?
2. **Second-order effects** — greedy allocation ignores module interactions. Should we do iterative refinement?
3. **Salient vs Fisher** — FABQ-RC uses Fisher. AWQ uses activation magnitude. Which matters more for EBQ's global allocation?
4. **KV cache coupling** — should KV precision affect weight allocation decisions (interactions via attention)?