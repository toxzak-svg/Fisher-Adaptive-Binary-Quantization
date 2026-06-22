from benchmark_fabq_runtime import ascii_preview, fabq_storage_bits, is_target_linear_name


def test_is_target_linear_name_excludes_embeddings_and_lm_head():
    assert not is_target_linear_name("model.embed_tokens")
    assert not is_target_linear_name("lm_head")


def test_is_target_linear_name_keeps_transformer_projections():
    assert is_target_linear_name("model.layers.0.self_attn.q_proj")
    assert is_target_linear_name("model.layers.0.mlp.gate_proj")


def test_fabq_storage_bits_counts_mixed_binary_int4_and_scales():
    bits = fabq_storage_bits(out_features=100, in_features=128, n_int4=5, blocksize=64)
    expected_int4 = 5 * 128 * 4
    expected_binary = 95 * 128
    expected_binary_scales = 95 * 2 * 16
    expected_int4_scales = 5 * 16
    expected_channel_map = 100 * 16
    assert bits == expected_int4 + expected_binary + expected_binary_scales + expected_int4_scales + expected_channel_map


def test_ascii_preview_escapes_non_ascii_for_windows_consoles():
    assert ascii_preview("abc Δ🚀", 20) == "abc \\u0394\\U0001f680"
