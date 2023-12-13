import torch
from transformers import LogitsProcessor, LogitsWarper

from watermarker.base import Base


class Processor(Base, LogitsWarper):
    """
    Processor for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        gamma: Penalty term, added to the green-list values
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = (self.strength * self.green_list_mask) + self.penalty
        new_logits = scores + watermark.to(scores.device)
        return new_logits