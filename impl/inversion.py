from typing import List, Optional

import diskcache
import torch
import vec2text
from numpy.typing import ArrayLike

from utils.globals import cache, inversion_embeddings_memory

_corrector = None
_device = None


@diskcache.barrier(cache, diskcache.Lock, expire=300)
def _invert_embeddings(
    embeddings: ArrayLike, num_steps: Optional[int], sequence_beam_width: Optional[int]
) -> List[str]:
    global _corrector, _device
    if _corrector is None:
        _corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

    if _device is None:
        _device = torch.device("cuda")

    embeddings = torch.from_numpy(embeddings).to(_device)
    return vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=_corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )


_cached_invert_embeddings = inversion_embeddings_memory.cache(_invert_embeddings)


def invert_embeddings(
    embeddings: ArrayLike, num_steps: Optional[int], sequence_beam_width: Optional[int]
) -> List[str]:
    return _cached_invert_embeddings(embeddings, num_steps, sequence_beam_width)
