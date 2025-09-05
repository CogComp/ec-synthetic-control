from collections.abc import Callable
from typing import List


type Embedding = Callable[[List[str]], List[List[float]]]
