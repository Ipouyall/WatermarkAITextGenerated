import torch
import numpy as np
import hashlib


# class Base:
#     def __init__(
#             self,
#             vocab: list,
#             hk=15485863,
#             cw=4,
#             salt=True
#     ):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         rng = torch.Generator(device=device)
#         rng.manual_seed(2971215073)
#         self.fixed_table = torch.randperm(1000003, device=device, generator=rng)
#         self.device = device
#
#     def _hashint(self, int_tensor: torch.LongTensor):
#         return self.fixed_table[int_tensor.to(self.device) % 1000003] + 1
#
#     def selfhash(self, input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
#         return (salt_key * self._hashint(input_ids) * self._hashint(input_ids[anchor])).min().item()


class Base:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        gamma: Penalty term, added to the green-list values
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(
            self,
            fraction: float = .5,
            strength: float = 2.,
            gamma: int = 0,
            vocab_size: int = 50257,
            watermark_key: int = 0):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        self.strength = strength
        self.fraction = fraction
        self.penalty = gamma

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

