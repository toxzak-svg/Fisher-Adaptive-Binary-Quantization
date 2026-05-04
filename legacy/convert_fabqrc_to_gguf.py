"""
Convert FABQ-RC quantized Mistral-7B to GGUF format.

FABQ-RC stores quantized weights as:
  - int8_channels: list of channel indices quantized to int8
  - int8_weights: (n_int8, in_channels) int8 weight values
  - int8_scales: (n_int8,) FP16 scale factors (1D tensor)
  - binary_channels: list of channel indices quantized to binary (±1)
  - binary_reconstructed_weights: (n_binary, in_channels) FP16 reconstructed

This script:
1. Loads the FABQ-RC quantized state dict
2. Reconstructs FP16 weights from quantized components
3. Writes to GGUF format
"""

import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType

# Model architecture for Mistral-7B
MISTRAL_ARCH = "mistral"


def load_fabqrc_state_dict(path: str):
    """Load FABQ-RC quantized state dict."""
    print(f"Loading FABQ-RC state dict from {path}...")
    state = torch.load(path, map_location='cpu', weights_only=False)
    print(f"  Loaded {len(state)} tensors")
    return state


def get_layer_info(state_dict):
    """Extract layer names and their quantized components."""
    layer_tensors = {}
    
    for key in state_dict.keys():
        if '.int8_channels' in key:
            layer_name = key.replace('.int8_channels', '')
            if layer_name not in layer_tensors:
                layer_tensors[layer_name] = {}
            layer_tensors[layer_name]['int8_channels'] = state_dict[key]
    
    return layer_tensors


def reconstruct_layer_fp16(state_dict, layer_name: str):
    """
    Reconstruct a single layer's FP16 weights from FABQ-RC quantized components.
    
    FABQ-RC structure per layer:
    - int8_channels: (n_int8,) list of channel indices using int8 quantization
    - int8_weights: (n_int8, in_channels) int8 weights
    - int8_scales: (n_int8,) FP16 scales (1D tensor)
    - binary_reconstructed_weights: (n_binary, in_channels) FP16 reconstructed weights
    
    Returns:
    - weight_fp16: (out_channels, in_channels) FP16 weight matrix
    """
    int8_channels = state_dict[f'{layer_name}.int8_channels']  # (n_int8,)
    int8_weights = state_dict[f'{layer_name}.int8_weights']    # (n_int8, in_channels)
    int8_scales = state_dict[f'{layer_name}.int8_scales']      # (n_int8,) - 1D tensor!
    binary_recon = state_dict[f'{layer_name}.binary_reconstructed_weights']  # (n_binary, in_channels)
    
    n_int8 = len(int8_channels)
    n_binary = binary_recon.shape[0]
    out_channels = n_int8 + n_binary
    in_channels = int8_weights.shape[1] if n_int8 > 0 else binary_recon.shape[1]
    
    # Reconstruct FP16 weight matrix
    weight = np.zeros((out_channels, in_channels), dtype=np.float32)
    
    # Reconstruct int8 channels
    int8_ch_list = int8_channels.tolist()
    for i, ch in enumerate(int8_ch_list):
        # int8_scales is 1D: int8_scales[i] gives scalar
        scale = int8_scales[i].item()
        weight[ch, :] = int8_weights[i].numpy().astype(np.float32) * scale
    
    # Reconstruct binary channels (they are stored directly as FP16)
    binary_idx = 0
    for c in range(out_channels):
        if c not in set(int8_ch_list):
            weight[c, :] = binary_recon[binary_idx].numpy()
            binary_idx += 1
    
    return weight


def get_gguf_tensor_name(fabq_name: str) -> str:
    """
    Map FABQ-RC layer names to GGUF tensor names.
    
    FABQ-RC uses patterns like:
    - model.layers.0.self_attn.q_proj
    - model.layers.0.mlp.gate_proj
    
    GGUF uses patterns like:
    - blk.0.attn_q.weight
    - blk.0.mlp.gate.weight
    """
    # Remove 'model.' prefix
    name = fabq_name.replace('model.', '')
    
    # Split into parts
    parts = name.split('.')
    
    if 'layers' in parts:
        # e.g., model.layers.0.self_attn.q_proj -> blk.0.attn_q.weight
        layer_idx = parts[parts.index('layers') + 1]
        
        # Find the component after layer number
        # e.g., self_attn -> attn, mlp -> mlp, gate_proj -> gate
        rest = parts[parts.index('layers') + 2:]
        
        if 'self_attn' in rest:
            attn_idx = rest.index('self_attn')
            proj = rest[attn_idx + 1]  # q_proj, k_proj, v_proj, o_proj
            proj_map = {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'o_proj': 'output'}
            tensor_type = f'attn_{proj_map.get(proj, proj)}'
        elif 'mlp' in rest:
            mlp_idx = rest.index('mlp')
            proj = rest[mlp_idx + 1]  # gate_proj, up_proj, down_proj
            proj_map = {'gate_proj': 'gate', 'up_proj': 'up', 'down_proj': 'down'}
            tensor_type = f'mlp.{proj_map.get(proj, proj)}'
        else:
            tensor_type = '.'.join(rest)
        
        return f'blk.{layer_idx}.{tensor_type}.weight'
    
    elif 'lm_head' in parts:
        return 'output.weight'
    
    elif 'embed_tokens' in parts:
        return 'token_embd.weight'
    
    return name + '.weight'


def add_mistral_metadata(writer):
    """Add Mistral-7B specific metadata."""
    writer.add_embedding_length(4096)
    writer.add_head_count(32)
    writer.add_head_count_kv(8)
    writer.add_block_count(32)
    writer.add_feed_forward_length(14336)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_context_length(32768)
    writer.add_rope_freq_base(1000000.0)
    writer.add_quantization_version(2)
    writer.add_tokenizer_model("llama")
    writer.add_vocab_size(32000)


def convert_fabqrc_to_gguf(fabqrc_path: str, output_path: str):
    """Convert FABQ-RC quantized model to GGUF format."""
    print("=" * 60)
    print("FABQ-RC to GGUF Conversion")
    print("=" * 60)
    
    # Load FABQ-RC state dict
    state_dict = load_fabqrc_state_dict(fabqrc_path)
    
    # Get FABQ-RC layer info
    layer_info = get_layer_info(state_dict)
    print(f"Found {len(layer_info)} quantized layers in FABQ-RC state dict")
    
    # Initialize GGUF writer
    print(f"\nInitializing GGUF writer with arch: {MISTRAL_ARCH}")
    writer = GGUFWriter(output_path, MISTRAL_ARCH, use_temp_file=False)
    
    # Add metadata
    print("Adding Mistral-7B metadata...")
    add_mistral_metadata(writer)
    
    print("\nReconstructing weights layer by layer...")
    total_params = 0
    written_tensors = 0
    
    for layer_name in layer_info.keys():
        try:
            weight_fp16 = reconstruct_layer_fp16(state_dict, layer_name)
            
            # Get GGUF tensor name
            tensor_name = get_gguf_tensor_name(layer_name)
            
            # Add tensor with proper shape specification
            writer.add_tensor(tensor_name, weight_fp16, raw_shape=weight_fp16.shape)
            
            total_params += weight_fp16.size
            written_tensors += 1
            
            if written_tensors % 20 == 0:
                print(f"  Processed {written_tensors} layers...")
                
        except Exception as e:
            print(f"  Warning: Could not process layer {layer_name}: {e}")
    
    print(f"\nWrote {written_tensors} tensors with {total_params:,} total parameters")
    
    # Write the GGUF file
    print(f"\nWriting GGUF file to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # Verify file was created
    import os
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"  File created successfully: {size:,} bytes")
    else:
        print("  WARNING: File was not created!")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Tensors: {written_tensors}")
    print(f"Parameters: {total_params:,}")
    print("\nNOTE: Weights are stored as FP16.")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert FABQ-RC quantized model to GGUF')
    parser.add_argument('input', help='FABQ-RC .pth file path')
    parser.add_argument('output', help='Output .gguf file path')
    
    args = parser.parse_args()
    
    convert_fabqrc_to_gguf(args.input, args.output)


if __name__ == '__main__':
    main()