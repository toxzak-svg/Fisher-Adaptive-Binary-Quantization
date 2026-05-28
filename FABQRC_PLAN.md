# FABQ-RC Research Plan

## Where We Are

- **Core method:** 1-bit quantization with adaptive per-layer blocksize + residual codebook
- **Key fix:** padded-block centroid bug (cells 19 and 21) — residuals computed on real data only, centroid skipped for padded blocks
- **Current results:** outperforms Q1_0_g128 and BiLLM at ~1.21 bpw (corrected from misleading ~1.15-1.20 claim)
- **Verified:** the bug exists and the fix is written; full validation pending

> **bpw accounting note:** The original spec claimed ~1.15-1.20 bpw but the codebook overhead compounds per block (uint8 index per 128 weights = 0.0625 bits/weight). With the architectural fixes (int4 top channels, tiered codebooks, 4-bit indices, min blocksize=64), the realistic actual bpw is ~1.21. The plan below reflects this corrected target.

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

### 2.1 Codebook architecture
- **Original:** single global 256-centroid codebook, uint8 indices
- **Ablate:** 4 tiered codebooks (64 centroids each, Fisher-quartile-based), 4-bit indices
- Measure quality vs compression tradeoff
- Expected: tiered codebooks cover residuals better at same storage cost

### 2.2 int4 vs int8 for top channels
- **Original:** int8 for top 5% channels (0.40 bits/weight)
- **Ablate:** int4 for top 5% channels (0.20 bits/weight)
- Measure quality degradation at 5% int4 vs int8
- Expected: minimal degradation since Fisher-identified channels are dense and well-distributed

### 2.3 Blocksize search range (corrected)
- **Original candidates:** {16, 32, 64, 128, 256}
- **Corrected candidates:** {64, 128, 256, 512} — dropped 16, 32
- Rationale: blocksize 16 inflates scale overhead to ~0.125 bits/weight vs ~0.03 at blocksize 64
- Test adaptive blocksize with different ceilings: 64, 128, 256, 512
- Check if optimal ceiling varies by model size or architecture type

### 2.4 Codebook index bit-width
- **Original:** uint8 (256 centroids accessible)
- **Ablate:** 4-bit (16 centroids accessible per layer cluster)
- Measure quality gap from reduced centroid access
- Expected: 16 centroids per cluster sufficient given tighter residual structure within Fisher tier

### 2.5 Blocksize selection algorithm
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
- **Report actual bpw (~1.21) not claimed bpw (~1.15-1.20)**

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
- **Accurate bpw reporting (actual ~1.21, not claimed ~1.15-1.20)**

### 5.3 Open source (optional)
- Clean up code for public release
- FABQ-RC GitHub (note: doesn't show up in search — worth fixing)
- License, README, example usage

---

## Priority Ordering

```
Week 1: Validation (1.1 → 1.2)
Week 2: Ablation studies (2.1 → 2.5) — includes new codebook/tier/ablation targets
Week 3: Baseline comparisons (3.1 → 3.2)
Week 4: Scale testing (4.1 → 4.3)
Ongoing: Documentation (5.1 → 5.3)
```

**Total estimated time:** 6-8 sessions depending on Colab access and how deep the ablations go.

---

## Cross-Cutting Concern: Calibration Robustness

> **Key insight from review:** The Fisher scores that drive all precision decisions depend entirely on the calibration distribution. A narrow calibration set (single domain, single sequence length) can produce "low Fisher" scores for channels that are genuinely important for rare facts, code/math, formatting, or long-range behavior. This risks turning adaptive quantization into task-specific pruning in disguise.

See `plans/CALIBRATION-ROBUSTNESS-PLAN.md` for the full treatment. The short version:

1. **Multi-domain Fisher**: Compute Fisher over C4 + code + math + long-context + instruction data, then aggregate via max (protect channels important in ANY domain)
2. **Domain-spike detection**: Promote channels that spike in a single domain even if their aggregated Fisher is low
3. **Context shape profiling**: Vary sequence length and position to detect "format-sensitive" channels
4. **Downstream flexibility**: Measure LoRA recovery cost, magnitude information loss, and adaptation-margin allocation
5. **Fisher diagnostics**: Bootstrap confidence intervals, cross-domain rank agreement (FSI), boundary channel audits

These are safeguards, not architectural changes. Stages 3 and 4 (adaptive blocksize, residual codebook) are untouched.

## Priority Rebalancing

Given the calibration concerns, the updated priority order is:

```
Week 1:   FABQ-RC Validation + Multi-domain Fisher infrastructure
Week 2:   Context shape profiling + Fisher stability diagnostics
Week 3:   Downstream flexibility analysis (LoRA recovery, magnitude loss)
Week 4:   Ablation studies (with robust calibration pipeline)
Week 5+:  Baseline comparisons, scale testing, documentation
```

## Immediate Next Step

Run FABQ_RC_Kaggle.ipynb with the corrected cells. Get before/after perplexity numbers. That's the only thing blocking everything else.
