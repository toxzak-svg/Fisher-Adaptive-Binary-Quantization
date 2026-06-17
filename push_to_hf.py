#!/usr/bin/env python3
"""push_to_hf.py - curate and push the fabq-rc project to HuggingFace.

Two repos:

  toxzak/fabq-rc-gemma4-12b   <- the new gemma4-12b/ folder, self-contained
  toxzak/fabq-rc              <- the parent project (curated, no debug noise)

The parent repo gets an explicit allowlist to keep the debug files (c20.txt,
cell*.txt, check_*.py, hs_err*.log, the legacy/ folder, etc.) out of HF.
"""

import os, sys, json, shutil
from pathlib import Path

# Where the two staged copies go before upload
STAGING = Path("./hf_staging")
Gemma_REPO = "toxzak/fabq-rc-gemma4-12b"
PARENT_REPO = "toxzak/fabq-rc"

# Source roots
SRC = Path(r"C:\Users\Zwmar\projects\fabq-rc")
GEMMA_SRC = SRC / "gemma4-12b"
PARENT_SRC = SRC

# -----------------------------------------------------------------------------
# Allowlist for the parent fabq-rc/ repo. Anything not in here is excluded.
# -----------------------------------------------------------------------------
# Top-level files / dirs we KEEP
PARENT_KEEP_TOP = {
    "README.md",
    "FABQ_RC_SPEC.md",
    "FABQRC_PLAN.md",
    "FABQ_RC_GGUF_SPEC.md",
    "MODEL_CARD.md",
    "MODEL_CARD_HF.md",
    "FABQ-RC-DeepSeek-V4-Flash.ipynb",
    "FABQ-RC-Dense-27B-Notebook.ipynb",
    "FABQ-RC-GGUF-Export.ipynb",
    "FABQ-RC-Phase0-Validation.ipynb",
    "FABQ-VP-8B-Notebook.ipynb",
    "Main-FABQ-RC-Notebook.ipynb",
    "gguf_pipeline.py",
    "export_fixed_gguf.py",
    "fix_gguf_reexport.py",
    "test_gguf_export.py",
    "minimal_gguf_test.py",
    "plans",
    "finetune",
    "notebooks",
    "latest notebooks",
    "legacy",  # keep the dir, contents excluded below
    "models",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
}

# Files to exclude even if they're at the top level (debug / one-off)
PARENT_EXCLUDE_TOP = {
    "c20.txt", "c20check.txt", "check20.txt", "final_check.txt",
    "cell1.txt", "cell13_debug.txt", "cell17.txt", "cell17_after.txt",
    "cell17_now.txt", "cell19_after.txt", "cell19_main.txt", "cell20.txt",
    "cells_28_50.txt", "cells_30_45.txt", "cells_33_to_end.txt",
    "cell_11_dense.txt", "cell_12_dense.txt", "cell_13_dense.txt",
    "cell_14_dense.txt", "cell_15_dense.txt", "cell_16_dense.txt",
    "cell_17_dense.txt", "cell_19_dense.txt", "cell_22.txt",
    "cell_22_dense.txt", "cell_23.txt", "cell_33.txt", "cell_34.txt",
    "main_30_45.txt", "main_cell_30.txt", "main_cell_31.txt",
    "main_cell_32.txt", "main_cell_33.txt", "main_cell_34.txt",
    "main_cell_35.txt", "main_cell_36.txt", "dense_28_43.txt",
    "dense_check.txt", "dup_check.txt", "final_FABQ_RC_Dense_27B_Notebook_ipynb.txt",
    "final_Main_FABQ_RC_Notebook_ipynb.txt", "clean_FABQ_RC_Dense_27B_Notebook_ipynb.txt",
    "clean_Main_FABQ_RC_Notebook_ipynb.txt", "check_FABQ-RC-Dense-27B-Notebook.ipynb_10.txt",
    "check_FABQ-RC-Dense-27B-Notebook.ipynb_11.txt",
    "check_FABQ-RC-Dense-27B-Notebook.ipynb_12.txt",
    "check_FABQ-RC-Dense-27B-Notebook.ipynb_13.txt",
    "check_FABQ-RC-Dense-27B-Notebook.ipynb_14.txt",
    "check_FABQ-RC-Dense-27B-Notebook.ipynb_15.txt", "nb_dump.txt",
    "summary_FABQ_RC_Dense_27B_Notebook_ipynb.txt",
    "summary_Main_FABQ_RC_Notebook_ipynb.txt", "stats.txt", "res.txt", "res2.txt",
    "inspect_output.txt",
    "hs_err_pid9332.log",
    "fabisdead",  # 13KB mystery file, name suggests it's a placeholder
    "fabqrc interestin.png",  # stray screenshot
    "check_cells.py", "check_cells_11_22.py", "check_dup.py",
    "check_main.py", "check_main_30_45.py", "check_now.py",
    "check_n_clusters.py", "check_tail.py",
    "clean_notebooks.py", "comp_clean.py", "create_fix_notebook.py",
    "debug_complete.py", "debug_gguf.py", "debug_layer.py", "debug_main.py",
    "debug_notebook.py", "debug_notebooks.py", "deep_clean.py",
    "deep_clean_notebook.py", "dump_middle.py", "final_check.py",
    "final_cleanup.py", "final_cleanup2.py", "fix_colab.py", "fix_int4_all.py",
    "fix_int4_refs.py", "fix_n_clusters.py", "fix_padding_bug.py",
    "fix_save.py", "inspect_int4.py", "inspect_weird.py", "kill_reload.py",
    "remove_all_standalone.py", "remove_bs.py", "remove_pure_imports.py",
    "strip_loading.py", "thorough_fix.py", "update_model.py",
    "update_nb.py", "update_nb2.py", "verify.py", "verify_cells_10_16.py",
    "verify_clean.py",
    "Copy_of_FABQ_RC_Quantization.ipynb",  # superseded by the clean Main
    ".idea",
    ".git",
    "__pycache__",
    "gemma4-12b",  # lives in its own repo
}

# legacy/ folder: keep the directory but replace contents with a README
LEGACY_README = """# legacy/

This folder contains earlier iterations of the FABQ-RC notebooks from April
and May 2026. They're kept here for reference but are NOT the current
working versions. The current working notebooks are:

- `../Main-FABQ-RC-Notebook.ipynb` - the Qwen3.6-27B baseline
- `../FABQ-RC-Dense-27B-Notebook.ipynb` - dense 27B experiments
- `../FABQ-RC-DeepSeek-V4-Flash.ipynb` - DeepSeek V4-Flash (MoE)
- `../FABQ-RC-GGUF-Export.ipynb` - GGUF export pipeline
- `../FABQ-RC-Phase0-Validation.ipynb` - validation phase
- `../FABQ-VP-8B-Notebook.ipynb` - FABQ-VP 8B variant

The Gemma 4 12B variant lives in `../gemma4-12b/` and is published
separately at https://huggingface.co/toxzak/fabq-rc-gemma4-12b.
"""


def stage_gemma_repo(gemma_dir: Path, dest: Path):
    """Copy the entire gemma4-12b/ folder to the staging dir, with HF metadata."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(gemma_dir, dest)

    # Write a metadata.json that tooling can pick up without parsing markdown
    metadata = {
        "license": "apache-2.0",
        "base_model": "google/gemma-4-12B-it",
        "tags": [
            "quantization", "1-bit", "fabq-rc", "fisher-adaptive",
            "gemma", "native-quantized-inference", "cuda-kernel", "code",
        ],
        "pipeline_tag": "other",
        "library_name": "fabq-rc",
        "project": "FABQ-RC",
        "description": (
            "FABQ-RC quantization pipeline for Gemma 4 12B-it. "
            "Two variants: text-only quantization (notebook) and streaming + "
            "native-quantized inference (custom CUDA kernel, no FP16 weight "
            "materialization at runtime)."
        ),
        "related_repos": [
            "toxzak/fabq-rc",
            "toxzak/gemma-4-12B-it-fabq-rc-bucket",
        ],
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Write a huggingface.yml for HF tooling (Spaces uses this, models
    # primarily use README frontmatter, but having both is harmless)
    hf_yml = """metadata:
  license: apache-2.0
  base_model: google/gemma-4-12B-it
  tags:
    - quantization
    - 1-bit
    - fabq-rc
    - fisher-adaptive
    - gemma
    - native-quantized-inference
    - cuda-kernel
    - code
  pipeline_tag: other
  library_name: fabq-rc
"""
    (dest / "huggingface.yml").write_text(hf_yml)

    # The README.md frontmatter is already in the source folder's README
    # (added by hand when this script was set up). Nothing to do here.


def stage_parent_repo(src: Path, dest: Path):
    """Curate the parent project: only the meaningful files, no debug noise."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy the keep-list
    for entry in PARENT_KEEP_TOP:
        src_path = src / entry
        if not src_path.exists():
            continue
        dest_path = dest / entry
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.pyc", "*.pyo",
                                ".cache", "*.bin", "*.safetensors",
                            ))
        else:
            shutil.copy2(src_path, dest_path)

    # Drop a README into legacy/ instead of its actual contents
    if (src / "legacy").exists():
        legacy_dest = dest / "legacy"
        if legacy_dest.exists():
            shutil.rmtree(legacy_dest)
        legacy_dest.mkdir(parents=True, exist_ok=True)
        (legacy_dest / "README.md").write_text(LEGACY_README)

    # Drop a .gitignore at the top
    (dest / ".gitignore").write_text("""__pycache__/
*.pyc
*.pyo
.cache/
*.bin
*.safetensors
.idea/
.ipynb_checkpoints/
""")

    # Write a metadata.json for the parent repo too
    metadata = {
        "license": "apache-2.0",
        "base_model": [
            "google/gemma-4-12B-it",
            "Qwen/Qwen3.6-27B",
            "deepseek-ai/DeepSeek-V4-Flash",
        ],
        "tags": ["quantization", "1-bit", "fabq-rc", "fisher-adaptive",
                 "research", "code"],
        "pipeline_tag": "other",
        "library_name": "fabq-rc",
        "project": "FABQ-RC",
        "description": (
            "Parent FABQ-RC project: working notebooks (Qwen 27B, Gemma 4 12B, "
            "DeepSeek V4-Flash), plans, specs, GGUF export scripts, finetune "
            "helpers. The Gemma 4 12B variant lives in a separate repo at "
            "toxzak/fabq-rc-gemma4-12b."
        ),
        "related_repos": [
            "toxzak/fabq-rc-gemma4-12b",
            "toxzak/gemma-4-12B-it-fabq-rc-bucket",
        ],
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # huggingface.yml for HF tooling
    hf_yml = """metadata:
  license: apache-2.0
  tags:
    - quantization
    - 1-bit
    - fabq-rc
    - fisher-adaptive
    - research
    - code
  pipeline_tag: other
  library_name: fabq-rc
"""
    (dest / "huggingface.yml").write_text(hf_yml)


def push_folder_to_hf(local_path: Path, repo_id: str, repo_type: str = "model",
                      token: str = None, commit_message: str = "Upload"):
    """Upload a local folder to a HF repo using the Hub API."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"  Creating repo {repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id, repo_type=repo_type, token=token,
                    exist_ok=True, private=False)
    print(f"  Uploading {local_path} to {repo_id}...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
        ignore_patterns=["__pycache__", "*.pyc", "*.pyo",
                         ".cache", "*.bin", "*.safetensors"],
    )
    print(f"  ✅ Pushed to https://huggingface.co/{repo_id}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Stage locally and show what would be pushed, but don't upload")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("❌ HF_TOKEN env var is not set. Aborting.", file=sys.stderr)
        sys.exit(1)

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    gemma_dest = STAGING / "fabq-rc-gemma4-12b"
    parent_dest = STAGING / "fabq-rc"

    print(f"=== Staging {gemma_dest.name} ===")
    stage_gemma_repo(GEMMA_SRC, gemma_dest)
    n_gemma = sum(1 for _ in gemma_dest.rglob("*") if _.is_file())
    print(f"  {n_gemma} files staged")

    print(f"\n=== Staging {parent_dest.name} (curated) ===")
    stage_parent_repo(PARENT_SRC, parent_dest)
    n_parent = sum(1 for _ in parent_dest.rglob("*") if _.is_file())
    print(f"  {n_parent} files staged")

    print(f"\n=== Pushing to HuggingFace ===")
    if args.dry_run:
        print("\n[DRY RUN] Not actually uploading. Inspect the staged folders at:")
        print(f"  {gemma_dest.absolute()}")
        print(f"  {parent_dest.absolute()}")
        print(f"\nRe-run without --dry-run to actually push.")
        return

    print(f"\n[1/2] Pushing gemma4-12b -> {Gemma_REPO}")
    push_folder_to_hf(gemma_dest, Gemma_REPO, "model", token,
                      commit_message="Initial upload: FABQ-RC for Gemma 4 12B-it")

    print(f"\n[2/2] Pushing parent fabq-rc -> {PARENT_REPO}")
    push_folder_to_hf(parent_dest, PARENT_REPO, "model", token,
                      commit_message="Initial curated upload of fabq-rc project")

    print(f"\n=== Done ===")
    print(f"  https://huggingface.co/{Gemma_REPO}")
    print(f"  https://huggingface.co/{PARENT_REPO}")


if __name__ == "__main__":
    main()
