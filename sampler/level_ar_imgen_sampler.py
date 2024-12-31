from .lm_sampler import top_k_top_p_filtering
def sample(
    model: Transformer,
    src,
    max_length=100,
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token=None,
    pad_token=None
):
    """
    Sample from the transformer model using various decoding strategies.
    
    Args:
        model: The transformer model
        src: Source input tensor of shape (batch_size, src_seq_len)
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (higher = more diverse)
        top_k: If set, only sample from the top k most likely tokens
        top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
        eos_token: End of sequence token ID
        pad_token: Padding token ID
    
    Returns:
        Generated sequence tensor of shape (batch_size, seq_len)
    """
    import torch
    import torch.nn.functional as F
    
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    # Encode the source sequence
    memory = model.encode(src)
    
    # Start with a tensor of just the start token
    output = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    # Initialize KV cache
    kv_cache = {i: {} for i in range(len(model.decoder_layers))}
    
    # Generate tokens one at a time
    for _ in range(max_length):
        # Generate logits for next token
        tgt_mask = model.make_tgt_mask(output)
        decoder_output, kv_cache = model.decode(
            output,
            memory,
            tgt_mask=tgt_mask,
            kv_cache=kv_cache
        )
        logits = model.fc_out(decoder_output[:, -1:])
        
        next_token_ids = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        # Append next token to output
        output = torch.cat([output, next_token_ids], dim=1)
        
        # Check if we should stop generation
        if eos_token is not None and torch.any(next_token_ids == eos_token).any():
            # Stop at first EOS token
            eos_positions = (output == eos_token).nonzero()
            if len(eos_positions) > 0:
                first_eos_pos = eos_positions[0][1]
                output = output[:, :first_eos_pos + 1]
                break
                
        # Check if we hit a padding token
        if pad_token is not None and torch.any(next_token_ids == pad_token).any():
            break
            
    return output