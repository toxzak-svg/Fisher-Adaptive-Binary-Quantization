# FABQ-RC Research Plan

## Where We Are

- **Core method:** 1-bit quantization with adaptive per-layer blocksize + residual codebook
- **Key fix:** padded-block centroid bug (cells 19 and 21) — residuals computed on real data only, centroid skipped for padded blocks
- **Current results:** outperforms Q1_0_g128 and BiLLM at ~1.18 bpw
- **Verified:** the bug exists and the fix is written; full validation pending

---

## Phase 1: Validation (1-2 sessions)

### 1.1 Confirm the fix works
- Run FABQ_RC_Kaggle.ipynb with the corrected cells (19, 21)
- Compare before/after cosine similarity on layer outputs
- Goal: all layers show cosine > 0.98

### 1.2 Measure quality impact
- Run perplexity benchmarks on the fixed version
- Compare against Q1_0_g128 and BiLLM baselines
- Confirm the improvement from the fix itself

**Deliverable:** validated baseline numbers to build on.

---

## Phase 2: Ablation Studies (2-3 sessions)

### 2.1 Codebook size
- Test codebook dimensions: 128, 256, 512, 1024
- Measure quality (perplexity) vs compression ratio tradeoff
- Expected: diminishing returns past 256 for most models

### 2.2 Blocksize search range
- Test adaptive blocksize with different ceilings: 64, 128, 256, 512
- Check if optimal ceiling varies by model size or architecture type

### 2.3 Residual codebook structure
- Vary residual codebook size independently
- Test: no residual codebook (baseline), 64, 128, 256 entries
- Does the residual component help more for larger models?

### 2.4 Blocksize selection algorithm
- Current: greedy search per layer
- Alternatives: uniform blocksize, layer-type-based fixed blocksize, clustering-based

**Deliverable:** sensitivity analysis showing which components actually matter.

---

## Phase 3: Baseline Comparisons (1-2 sessions)

### 3.1 Run against available baselines
- Q1_0_g128 (already done)
- BiLLM (already done)
- If possible: QuIP#2, AQLM, LUT-based 1-bit

### 3.2 Collect standardized numbers
- Same model architecture across all methods
- Same benchmark dataset (Wikitext, C4, etc.)
- Same bits-per-parameter across all methods

**Deliverable:** comparison table showing FABQ-RC vs alternatives across quality/compression.

---

## Phase 4: Scale Testing (2-3 sessions)

### 4.1 Smaller models (confirm advantage holds)
- Run on 7B scale (if accessible locally, or via Colab)
- Compare against Q1_0_g128 and BiLLM at same bpw

### 4.2 Medium models (13B, 14B)
- Colab or accessible GPU
- Same benchmark suite
- Test if advantage scales, stays flat, or degrades

### 4.3 Larger models (if feasible)
- 70B+ scale
- Check if per-layer adaptive blocksize becomes more or less important at scale

**Deliverable:** scaling curve showing how FABQ-RC performs as model size grows.

---

## Phase 5: Documentation (ongoing)

### 5.1 Write up the method
- Clear explanation of adaptive blocksize mechanism
- Residual codebook role and why it works
- How the fix addresses the padded-block centroid bug

### 5.2 Results summary
- Ablation results in digestible format
- Comparison table with baselines
- Scaling behavior

### 5.3 Open source (optional)
- Clean up code for public release
- FABQ-RC GitHub (note: doesn't show up in search — worth fixing)
- License, README, example usage

---

## Priority Ordering

```
Week 1: Validation (1.1 → 1.2)
Week 2: Ablation studies (2.1 → 2.4)
Week 3: Baseline comparisons (3.1 → 3.2)
Week 4: Scale testing (4.1 → 4.3)
Ongoing: Documentation (5.1 → 5.3)
```

**Total estimated time:** 6-8 sessions depending on Colab access and how deep the ablations go.

---

## Immediate Next Step

Run FABQ_RC_Kaggle.ipynb with the corrected cells. Get before/after perplexity numbers. That's the only thing blocking everything else.
