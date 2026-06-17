# FABQ-RC Gemma 4 12B - Makefile
#
# One-command surface area for the whole project:
#   make install         install Python deps
#   make test-cuda       build the CUDA extension and run numerical tests
#   make bucket          one-time: populate the HF bucket
#   make notebook-text   launch the simple text quantization notebook
#   make notebook-stream   launch the streaming + native-quantized notebook
#   make push            re-push both HF repos
#   make clean           remove staging dir + extension build artifacts
#
# All targets assume you're in the gemma4-12b/ folder.
#
# Required environment:
#   HF_TOKEN       HuggingFace token (https://huggingface.co/settings/tokens)
#                  The gemma-4-12B-it model is gated; you need read access.
#   CUDA_HOME      Path to your CUDA toolkit install (for the CUDA extension).
#                  Not required for `install`, `notebook-text`, or `push`.
#
# On Windows, run with:  make <target>
# (MinGW / Git Bash) or:  mingw32-make <target>
# PowerShell can use make.exe if installed (via chocolatey or Visual Studio).

# ---- Config ----
PYTHON      ?= python
HF_TOKEN    ?= $(shell echo $$HF_TOKEN)
HF_BUCKET   ?= toxzak/gemma-4-12B-it-fabq-rc-bucket
GEMMA_REPO  ?= toxzak/fabq-rc-gemma4-12b
PARENT_REPO ?= toxzak/fabq-rc
SOURCE_MODEL ?= google/gemma-4-12B-it
EXT_DIR     := streaming/fabq_rc_cuda

# Auto-load .env if present
ifneq (,$(wildcard .env))
include .env
export
endif

# ---- Phony targets ----
.PHONY: help install test-cuda bucket notebook-text notebook-stream \
        push push-dry-run clean check-env

help:  ## Show this help
	@echo "FABQ-RC Gemma 4 12B - available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Required env vars:"
	@echo "  HF_TOKEN     - HuggingFace token (gated model access)"
	@echo "  CUDA_HOME    - CUDA toolkit path (only for test-cuda / bucket)"

# ---- Setup ----

install:  ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ Install complete."
	@echo "   Next: set HF_TOKEN and (optionally) CUDA_HOME, then 'make test-cuda'"

check-env:  ## Verify required environment variables
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "❌ HF_TOKEN is not set. Get a token at https://huggingface.co/settings/tokens"; \
		echo "   and either: export HF_TOKEN=hf_...  (Linux/Mac)"; \
		echo "   or:       setx HF_TOKEN hf_...    (Windows)"; \
		exit 1; \
	else \
		echo "✅ HF_TOKEN is set (length: $(shell echo -n '$(HF_TOKEN)' | wc -c))"; \
	fi
	@if [ -z "$(CUDA_HOME)" ]; then \
		echo "⚠️  CUDA_HOME is not set. You need it for test-cuda and bucket."; \
		echo "   On Linux/Mac:  export CUDA_HOME=/usr/local/cuda"; \
		echo "   On Windows:    setx CUDA_HOME \"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\""; \
	else \
		echo "✅ CUDA_HOME = $(CUDA_HOME)"; \
	fi

# ---- CUDA extension ----

test-cuda: check-env  ## Build the fabq_rc_cuda extension and run numerical tests
	@echo "🔨 Building fabq_rc_cuda extension (first run, 2-5 min)..."
	cd $(EXT_DIR) && $(PYTHON) setup.py build_ext --inplace
	@echo ""
	@echo "🧪 Running numerical tests..."
	cd $(EXT_DIR) && $(PYTHON) tests/test_kernel.py
	@echo ""
	@echo "✅ CUDA extension built and verified."

# ---- Bucket (one-time) ----

bucket: check-env  ## One-time: populate the HF bucket with pre-quantized shards (~30-45 min on A100)
	cd streaming && $(PYTHON) build_bucket.py --source $(SOURCE_MODEL) --push
	@echo ""
	@echo "✅ Bucket populated: https://huggingface.co/$(HF_BUCKET)"
	@echo "   You can now run 'make notebook-stream'."

# ---- Notebooks ----

notebook-text: check-env  ## Launch the text-only quantization notebook
	@echo "📓 Launching FABQ-RC-Gemma4-12B.ipynb..."
	@echo "   Hardware: needs A100 80GB."
	@echo "   Runtime:  ~30-60 min."
	$(PYTHON) -m jupyter notebook FABQ-RC-Gemma4-12B.ipynb

notebook-stream: check-env  ## Launch the streaming + native-quantized inference notebook
	@echo "📓 Launching FABQ-RC-Gemma4-12B-Streaming.ipynb..."
	@echo "   Hardware: needs A100 80GB + CUDA toolchain."
	@echo "   Prereq:   bucket must be populated ('make bucket')."
	@echo "   Runtime:  ~5-10 min (mostly the in-cell CUDA build on first run)."
	cd streaming && $(PYTHON) -m jupyter notebook FABQ-RC-Gemma4-12B-Streaming.ipynb

# ---- HF push ----

push:  ## Re-push both HF repos (curated)
	$(PYTHON) push_to_hf.py
	@echo ""
	@echo "✅ Pushed to:"
	@echo "   https://huggingface.co/$(GEMMA_REPO)"
	@echo "   https://huggingface.co/$(PARENT_REPO)"

push-dry-run:  ## Show what would be pushed without actually uploading
	$(PYTHON) push_to_hf.py --dry-run

# ---- Cleanup ----

clean:  ## Remove staging dir + built extension artifacts
	rm -rf hf_staging
	cd $(EXT_DIR) && rm -rf build/ *.so
	@echo "✅ Cleaned."
