import math

from benchmark_qwen35_runtime import aggregate_loss, auto_model_kind, tokens_per_second


def test_tokens_per_second_counts_generated_tokens_only():
    assert tokens_per_second(prompt_tokens=5, total_tokens=13, elapsed_sec=2.0) == 4.0


def test_tokens_per_second_returns_zero_for_nonpositive_elapsed():
    assert tokens_per_second(prompt_tokens=5, total_tokens=13, elapsed_sec=0.0) == 0.0


def test_aggregate_loss_returns_perplexity_from_weighted_losses():
    result = aggregate_loss([(math.log(2.0), 10), (math.log(4.0), 10)])
    assert result["tokens"] == 20
    assert abs(result["loss"] - math.log(math.sqrt(8.0))) < 1e-12
    assert abs(result["perplexity"] - math.sqrt(8.0)) < 1e-12


def test_auto_model_kind_uses_multimodal_for_conditional_generation():
    assert auto_model_kind("qwen3_5", ["Qwen3_5ForConditionalGeneration"]) == "image_text_to_text"


def test_auto_model_kind_uses_causal_lm_for_text_generation():
    assert auto_model_kind("qwen3", ["Qwen3ForCausalLM"]) == "causal_lm"
