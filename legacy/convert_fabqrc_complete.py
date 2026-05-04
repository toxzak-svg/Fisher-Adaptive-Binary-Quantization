"""
Complete GGUF conversion for FABQ-RC quantized Mistral-7B.
Downloads tokenizer from HuggingFace and creates a loadable GGUF.
"""

import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter
import os
import tempfile

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
    """Reconstruct FP16 weights from FABQ-RC components."""
    int8_channels = state_dict[f'{layer_name}.int8_channels']
    int8_weights = state_dict[f'{layer_name}.int8_weights']
    int8_scales = state_dict[f'{layer_name}.int8_scales']
    binary_recon = state_dict[f'{layer_name}.binary_reconstructed_weights']
    
    n_int8 = len(int8_channels)
    n_binary = binary_recon.shape[0]
    out_channels = n_int8 + n_binary
    in_channels = int8_weights.shape[1] if n_int8 > 0 else binary_recon.shape[1]
    
    weight = np.zeros((out_channels, in_channels), dtype=np.float32)
    
    int8_ch_list = int8_channels.tolist()
    for i, ch in enumerate(int8_ch_list):
        scale = int8_scales[i].item()
        weight[ch, :] = int8_weights[i].numpy().astype(np.float32) * scale
    
    binary_idx = 0
    for c in range(out_channels):
        if c not in set(int8_ch_list):
            weight[c, :] = binary_recon[binary_idx].numpy()
            binary_idx += 1
    
    return weight


def get_gguf_tensor_name(fabq_name: str) -> str:
    """Map FABQ-RC layer names to GGUF tensor names."""
    name = fabq_name.replace('model.', '')
    parts = name.split('.')
    
    if 'layers' in parts:
        layer_idx = parts[parts.index('layers') + 1]
        rest = parts[parts.index('layers') + 2:]
        
        if 'self_attn' in rest:
            attn_idx = rest.index('self_attn')
            proj = rest[attn_idx + 1]
            proj_map = {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'o_proj': 'output'}
            tensor_type = f'attn_{proj_map.get(proj, proj)}'
        elif 'mlp' in rest:
            mlp_idx = rest.index('mlp')
            proj = rest[mlp_idx + 1]
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


def download_tokenizer():
    """Download tokenizer files from HuggingFace Mistral-7B."""
    from huggingface_hub import snapshot_download
    
    print("Downloading tokenizer from HuggingFace...")
    
    # Create temp directory for tokenizer files
    temp_dir = tempfile.mkdtemp()
    
    # Download only tokenizer files (not the model weights)
    snapshot_download(
        repo_id="mistralai/Mistral-7B-v0.1",
        local_dir=temp_dir,
        allow_patterns=["tokenizer*", "*.json", "*.txt", "*.model"],
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"]
    )
    
    print(f"Tokenizer downloaded to: {temp_dir}")
    return temp_dir


def find_tokenizer_files(tokenizer_dir):
    """Find tokenizer files in the downloaded directory."""
    files = os.listdir(tokenizer_dir)
    tokenizer_files = {}
    
    for f in files:
        if 'tokenizer' in f.lower():
            tokenizer_files[f] = os.path.join(tokenizer_dir, f)
    
    print(f"Found tokenizer files: {list(tokenizer_files.keys())}")
    return tokenizer_files


def add_tokenizer_to_gguf(writer, tokenizer_dir):
    """Add tokenizer data to GGUF writer."""
    tokenizer_files = find_tokenizer_files(tokenizer_dir)
    
    # Find the main tokenizer config/file
    config_file = None
    for fname in ['tokenizer_config.json', 'tokenizer.json']:
        if fname in tokenizer_files:
            config_file = tokenizer_files[fname]
            break
    
    if config_file:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Add tokenizer model type
        tokenizer_model = config.get('model_type', 'llama')
        writer.add_tokenizer_model(tokenizer_model)
        
        # Add vocab size
        if 'vocab_size' in config:
            writer.add_vocab_size(config['vocab_size'])
        elif 'added_tokens_decoder' in config:
            writer.add_vocab_size(len(config['added_tokens_decoder']))
        else:
            writer.add_vocab_size(32000)  # Default for Mistral
        
        # Try to add token list from tokenizer.json
        if 'tokenizer.json' in tokenizer_files:
            try:
                with open(tokenizer_files['tokenizer.json'], 'r', encoding='utf-8') as f:
                    tok_data = json.load(f)
                
                if 'model' in tok_data and 'vocab' in tok_data['model']:
                    vocab = tok_data['model']['vocab']
                    # vocab is typically {id: token_string} or {token_string: score}
                    if isinstance(vocab, dict):
                        # Check if values are strings (id->token) or numbers (token->score)
                        sample_val = next(iter(vocab.values())) if vocab else None
                        if isinstance(sample_val, str):
                            # Format: {id: token_string}
                            tokens = [vocab[i] for i in sorted(vocab.keys())]
                            scores = [0.0] * len(tokens)
                        else:
                            # Format: {token_string: score}
                            tokens = list(vocab.keys())
                            scores = list(vocab.values())
                        
                        writer.add_token_list(tokens)
                        writer.add_token_scores(scores)
            except (UnicodeDecodeError, KeyError, json.JSONDecodeError) as e:
                print(f"  Warning: Could not read tokenizer.json ({type(e).__name__}), skipping token list")
                pass
        
        # Add special tokens if available
        if 'added_tokens_decoder' in config:
            for token_data in config['added_tokens_decoder'].values():
                if token_data.get('special'):
                    if token_data.get('bos'):
                        writer.add_add_bos_token(True)
                    if token_data.get('eos'):
                        writer.add_add_eos_token(True)
    else:
        # Fallback: just add basic tokenizer config
        writer.add_tokenizer_model("llama")
        writer.add_vocab_size(32000)
        writer.add_add_bos_token(True)
        writer.add_add_eos_token(True)


def convert_fabqrc_to_complete_gguf(fabqrc_path: str, output_path: str):
    """Convert FABQ-RC model with complete tokenizer."""
    print("=" * 60)
    print("FABQ-RC to GGUF Conversion (with Tokenizer)")
    print("=" * 60)
    
    # Load FABQ-RC state dict
    state_dict = load_fabqrc_state_dict(fabqrc_path)
    
    # Get FABQ-RC layer info
    layer_info = get_layer_info(state_dict)
    print(f"Found {len(layer_info)} quantized layers in FABQ-RC state dict")
    
    # Download tokenizer
    tokenizer_dir = download_tokenizer()
    
    # Initialize GGUF writer
    print(f"\nInitializing GGUF writer with arch: {MISTRAL_ARCH}")
    writer = GGUFWriter(output_path, MISTRAL_ARCH, use_temp_file=False)
    
    # Add metadata
    print("Adding Mistral-7B metadata...")
    writer.add_embedding_length(4096)
    writer.add_head_count(32)
    writer.add_head_count_kv(8)
    writer.add_block_count(32)
    writer.add_feed_forward_length(14336)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_context_length(32768)
    writer.add_rope_freq_base(1000000.0)
    writer.add_quantization_version(2)
    
    # Add tokenizer
    print("Adding tokenizer...")
    add_tokenizer_to_gguf(writer, tokenizer_dir)
    
    print("\nReconstructing weights layer by layer...")
    total_params = 0
    written_tensors = 0
    
    for layer_name in layer_info.keys():
        try:
            weight_fp16 = reconstruct_layer_fp16(state_dict, layer_name)
            tensor_name = get_gguf_tensor_name(layer_name)
            writer.add_tensor(tensor_name, weight_fp16, raw_shape=weight_fp16.shape)
            
            total_params += weight_fp16.size
            written_tensors += 1
            
            if written_tensors % 20 == 0:
                print(f"  Processed {written_tensors} layers...")
                
        except Exception as e:
            print(f"  Warning: Could not process layer {layer_name}: {e}")
    
    print(f"\nWrote {written_tensors} tensors with {total_params:,} total parameters")
    
    # Write GGUF
    print(f"\nWriting GGUF file to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # Cleanup
    import shutil
    shutil.rmtree(tokenizer_dir, ignore_errors=True)
    
    # Verify
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"\n✅ File created successfully: {size:,} bytes")
    else:
        print("\n❌ File was NOT created!")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Tensors: {written_tensors}")
    print(f"Parameters: {total_params:,}")
    print(f"Note: Weights stored as FP16, tokenizer added from HuggingFace")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert FABQ-RC to complete GGUF with tokenizer')
    parser.add_argument('input', help='FABQ-RC .pth file path')
    parser.add_argument('output', help='Output .gguf file path')
    
    args = parser.parse_args()
    
    convert_fabqrc_to_complete_gguf(args.input, args.output)


if __name__ == '__main__':
    main()