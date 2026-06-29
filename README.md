# FABQ-RC

Fisher-Adaptive Binary Quantization with Residual Codebooks.

This repository is an active research prototype for sub-byte LLM
quantization. Treat the current state as a validated prototype plus known
gaps, not as a final benchmark release. The strongest checked-in evidence is
in `results/`, `docs/validation/VALIDATION_MEMO.md`, and `paper/`.

## Current Status

- Method and format specs are organized under `docs/specs/`.
- Validation notes live under `docs/validation/`.
- Measured benchmark and runtime artifacts live under `results/`.
- Paper and Substack drafts live under `paper/`.
- Historical notebook repair/extraction material is preserved under
  `artifacts/` and `scripts/notebook-maintenance/`.

Key caveat: older `~1.18 bpw` and final `27B` claims are not fully validated.
Use the validation memo and result files as the source of truth before making
publication or model-card claims.

## Repository Layout

```text
fabq-rc/
  README.md                    Project map and current status
  CHANGELOG.md                 Sync history
  benchmarks/                  Benchmark runners and helper tests
  docs/
    specs/                     Core method and GGUF specs
    validation/                Validation memo and claim audit
    model-cards/               Draft model-card material
    notes/                     Loose research notes
  paper/                       Preprint and Substack publication drafts
  plans/                       Research plans and extension specs
  results/                     Benchmark JSON, logs, and reports
  notebooks/
    archive/                   Root notebooks moved out of the repo root
    latest/                    Former `latest notebooks/` contents
  scripts/
    gguf/                      GGUF export and smoke-test scripts
    notebook-maintenance/      One-off notebook cleaning/debug helpers
    publishing/                Hugging Face staging/publishing helper
  artifacts/
    notebook-extractions/      Extracted notebook cells and summaries
    scratch/                   Tiny scratch outputs
  assets/images/               Images used by docs or notes
  logs/                        Local crash logs and runtime logs
  gemma4-12b/                  Gemma 4 12B variant and streaming runtime
  finetune/                    Finetuning helper
  legacy/                      Older project material
  models/                      Local model cache/placeholders
```

## Main Entry Points

- `docs/specs/FABQ_RC_SPEC.md` - primary FABQ-RC method specification.
- `docs/specs/FABQ_RC_GGUF_SPEC.md` - root GGUF format draft.
- `docs/validation/VALIDATION_MEMO.md` - claim audit and validation gaps.
- `results/qwen35_08b_weight_quant.md` - measured Qwen3.5 0.8B weight
  quantization summary.
- `results/runtime_validation_report.md` and
  `results/fabq_runtime_validation_report.md` - runtime validation reports.
- `paper/FABQ_RC_preprint.md` - conservative technical-report draft.
- `paper/substack/README.md` - three-post reader-facing publication series.
- `notebooks/archive/Main-FABQ-RC-Notebook.ipynb` - archived main notebook.
- `notebooks/archive/FABQ-RC-Dense-27B-Notebook.ipynb` - archived dense 27B
  experiment notebook.
- `scripts/publishing/push_to_hf.py` - curated Hugging Face staging helper.

## Validation Guidance

Before presenting FABQ-RC as a release-ready result, check:

1. Storage accounting against `docs/validation/VALIDATION_MEMO.md`.
2. Perplexity and runtime numbers in `results/`.
3. Whether the padded-centroid issue in `plans/RESEARCH-PLAN.md` is verified.
4. Whether the GGUF spec being referenced is the intended canonical version.

The project has useful negative and prototype results, but the README and
model-card claims should stay conservative until those checks are complete.

## License

Apache 2.0.
