import torch

from model import encoder_decoder_transformer
from .lm_sampler import top_k_top_p_filtering
from .lm_sampler import GenerationConfig

class LevelARImGenSampler:
    def __init__(self, models, vqvae):
        self.models = models
        self.vqvae = vqvae

    def sample_level_token_ids(
        self,
        condition: torch.LongTensor,
        code_length: int,
        level: int,
        config: GenerationConfig = None
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
        
        model = self.models[level]
        model: encoder_decoder_transformer.Transformer

        device = condition.device
        batch_size = condition.size(0)
        
        # Encode the source sequence
        memory = model.encode(condition)
        
        # Start with a tensor of just the start token
        output = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        # Initialize KV cache
        kv_cache = {i: {} for i in range(len(model.decoder_layers))}
        
        # Generate tokens one at a time
        for _ in range(code_length):
            # Generate logits for next token
            tgt_mask = model.make_tgt_mask(output)
            decoder_output, kv_cache = model.decode(
                output,
                memory,
                tgt_mask=tgt_mask,
                kv_cache=kv_cache
            )
            logits = model.fc_out(decoder_output[:, -1:])
            
            next_token_ids = top_k_top_p_filtering(
                logits / config.temperature, top_k=0, top_p=config.top_p
            )
            
            # Append next token to output
            output = torch.cat([output, next_token_ids], dim=1)
            
        return output

    def decode_token_ids(self, token_ids: list[torch.LongTensor]):
        """
        Decode a tensor of token IDs into a list of strings.
        
        Args:
            token_ids: Tensor of token IDs of shape (batch_size, seq_len)
        
        Returns:
            List of strings
        """
        return self.vqvae.decode_codes(token_ids)

    def sample_image(self, label):
        """
        Sample an image from the VQ-VAE model.
        
        Args:
            label: Label tensor of shape (batch_size, 1)
        
        Returns:
            Image tensor of shape (batch_size, 3, 128, 128)
        """